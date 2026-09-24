# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _entropy_route_teacher_signal(
    teacher_log_probs: torch.Tensor,
    teacher_entropies: torch.Tensor | None,
    temperature: float = 0.1,
    aggregation: str = "entropy",
    teacher_utilities: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate sampled-token signals from a token-aligned teacher pool.

    Args:
        teacher_log_probs: ``[num_teachers, batch, response_length]``.
        teacher_entropies: Tensor with the same shape.
        temperature: Softmax temperature for Entropy and Novelty routing.
            Ignored by the ``mean`` aggregation baseline.
        aggregation: ``entropy`` minimizes predictive entropy,
            ``accessible_novelty`` maximizes the supplied intrinsic utility,
            and ``mean`` equally averages all teachers.
        teacher_utilities: Required for ``accessible_novelty`` and shaped like
            ``teacher_log_probs``.

    Returns:
        The routed log probability ``[batch, response_length]`` and routing
        weights ``[num_teachers, batch, response_length]``.
    """
    if teacher_log_probs.ndim != 3 or teacher_log_probs.shape[0] < 2:
        raise ValueError(
            "Entropy-aware routing requires [num_teachers, batch, response_length] "
            "tensors with at least two teachers."
        )
    if aggregation != "mean" and temperature <= 0:
        raise ValueError(f"router temperature must be positive, got {temperature}.")
    if aggregation not in {"entropy", "mean", "accessible_novelty"}:
        raise ValueError(
            "teacher_signal_aggregation must be one of "
            "{'entropy', 'mean', 'accessible_novelty'}, "
            f"got {aggregation!r}."
        )
    if aggregation == "entropy" and (
        teacher_entropies is None or teacher_log_probs.shape != teacher_entropies.shape
    ):
        entropy_shape = None if teacher_entropies is None else teacher_entropies.shape
        raise ValueError(
            "Teacher log-prob and entropy tensors must have identical shapes, "
            f"got {teacher_log_probs.shape} and {entropy_shape}."
        )

    if aggregation == "mean":
        weights = torch.full_like(
            teacher_log_probs.detach().float(), 1.0 / teacher_log_probs.shape[0]
        )
    elif aggregation == "accessible_novelty":
        if teacher_utilities is None:
            raise ValueError("accessible_novelty routing requires teacher_utilities")
        if teacher_utilities.shape != teacher_log_probs.shape:
            raise ValueError(
                "teacher_utilities must match teacher_log_probs, got "
                f"{teacher_utilities.shape} and {teacher_log_probs.shape}."
            )
        utilities = teacher_utilities.detach().float()
        weights = torch.softmax(utilities / temperature, dim=0)
    else:
        entropies = teacher_entropies.detach().float()
        weights = torch.softmax(-entropies / temperature, dim=0)

    routed_log_probs = (weights.to(teacher_log_probs.dtype) * teacher_log_probs).sum(dim=0)
    return routed_log_probs, weights


def _delta_route_teacher_signal(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_base_topk_log_probs: torch.Tensor,
    weighting: str = "cosine",
    domain_teacher_indices: torch.Tensor | None = None,
    response_mask: torch.Tensor | None = None,
    alignment_epsilon: float = 1e-6,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route teachers by aligned specialization and teaching directions.

    All vectors are evaluated on the same student top-k support.  The
    teacher-base vector represents the teacher's specialization direction,
    while the teacher-student vector represents its current teaching
    direction. Teachers with a non-positive inner product are filtered; the
    remaining teachers are combined using either uniform or normalized cosine
    weights. Domain-only restricts the eligible pool, while rollout-level
    reuses one teacher mixture throughout a response.

    Args:
        student_topk_log_probs: ``[batch, response_length, topk]``.
        teacher_topk_log_probs: ``[num_teachers, batch, response_length, topk]``.
        teacher_base_topk_log_probs: Same shape as ``student_topk_log_probs``.
        weighting: One of ``uniform``, ``cosine``, ``domain_only``, or
            ``rollout_level``. ``cosine`` normalizes positive alignment over
            the retained set.
            ``domain_only`` retains only the sample's labeled-domain teacher
            after the same positive-alignment gate.
            ``rollout_level`` averages each teacher's non-negative cosine over
            valid response tokens, normalizes across teachers, and reuses the
            resulting fixed weights at every token in that rollout.
        domain_teacher_indices: ``[batch]`` teacher indices required by
            ``domain_only``. Ignored by other weighting rules.
        response_mask: ``[batch, response_length]`` valid-token mask required
            by ``rollout_level``. Ignored by other weighting rules.
        alignment_epsilon: A teacher is retained only when the raw inner
            product of its specialization and teaching vectors exceeds this
            non-negative margin.
        eps: Numerical floor for vector norms and cosine denominators.

    Returns:
        Routing coefficients, cosine alignments, and specialization norms,
        each with shape ``[num_teachers, batch, response_length]``. The
        coefficients sum to one when at least one eligible teacher survives.
    """
    if teacher_topk_log_probs.ndim != 4 or teacher_topk_log_probs.shape[0] < 2:
        raise ValueError(
            "Delta routing requires teacher_topk_log_probs with shape "
            "[num_teachers, batch, response_length, topk] and at least two teachers."
        )
    expected_shape = teacher_topk_log_probs.shape[1:]
    if student_topk_log_probs.shape != expected_shape:
        raise ValueError(
            "student_topk_log_probs must match the non-teacher dimensions of "
            f"teacher_topk_log_probs, got {student_topk_log_probs.shape} and {expected_shape}."
        )
    if teacher_base_topk_log_probs.shape != expected_shape:
        raise ValueError(
            "teacher_base_topk_log_probs must match student_topk_log_probs, got "
            f"{teacher_base_topk_log_probs.shape} and {student_topk_log_probs.shape}."
        )
    valid_weightings = {
        "uniform",
        "cosine",
        "domain_only",
        "rollout_level",
    }
    if weighting not in valid_weightings:
        raise ValueError(
            f"delta_router_weighting must be one of {sorted(valid_weightings)}, got {weighting!r}."
        )
    if weighting == "domain_only":
        if domain_teacher_indices is None:
            raise ValueError("delta_router_weighting='domain_only' requires domain_teacher_indices.")
        if domain_teacher_indices.ndim != 1 or domain_teacher_indices.shape[0] != expected_shape[0]:
            raise ValueError(
                "domain_teacher_indices must have shape [batch], got "
                f"{domain_teacher_indices.shape} for batch size {expected_shape[0]}."
            )
        if (
            (domain_teacher_indices < 0).any()
            or (domain_teacher_indices >= teacher_topk_log_probs.shape[0]).any()
        ):
            raise ValueError(
                "domain_teacher_indices contains an index outside the configured teacher pool."
            )
    if weighting == "rollout_level":
        if response_mask is None:
            raise ValueError("delta_router_weighting='rollout_level' requires response_mask.")
        if response_mask.shape != expected_shape[:2]:
            raise ValueError(
                "response_mask must have shape [batch, response_length], got "
                f"{response_mask.shape} for expected shape {expected_shape[:2]}."
            )
    if alignment_epsilon < 0:
        raise ValueError(
            "delta_router_alignment_epsilon must be non-negative, "
            f"got {alignment_epsilon}."
        )

    student_log_probs = student_topk_log_probs.detach().float()
    teacher_log_probs = teacher_topk_log_probs.detach().float()
    base_log_probs = teacher_base_topk_log_probs.detach().float()

    specialization = teacher_log_probs - base_log_probs.unsqueeze(0)
    teaching = teacher_log_probs - student_log_probs.unsqueeze(0)

    specialization_sq_norm = specialization.square().sum(dim=-1)
    teaching_sq_norm = teaching.square().sum(dim=-1)
    specialization_norm = specialization_sq_norm.clamp_min(0).sqrt()
    teaching_norm = teaching_sq_norm.clamp_min(0).sqrt()
    inner_product = (specialization * teaching).sum(dim=-1)

    valid = (
        (inner_product > alignment_epsilon)
        & (specialization_norm > eps)
        & (teaching_norm > eps)
    )
    cosine = inner_product / (specialization_norm * teaching_norm).clamp_min(eps)
    cosine = torch.where(valid, cosine, torch.zeros_like(cosine))

    if weighting == "uniform":
        scores = valid.to(specialization_norm.dtype)
        weights = scores / scores.sum(dim=0, keepdim=True).clamp_min(1.0)
    elif weighting == "cosine":
        scores = cosine
        weights = scores / scores.sum(dim=0, keepdim=True).clamp_min(eps)
    elif weighting == "domain_only":
        domain_indices = domain_teacher_indices.to(device=cosine.device, dtype=torch.long)
        domain_mask = torch.zeros_like(valid)
        scatter_index = domain_indices.view(1, -1, 1).expand(1, -1, valid.shape[-1])
        domain_mask.scatter_(0, scatter_index, True)
        scores = cosine * domain_mask
        weights = scores / scores.sum(dim=0, keepdim=True).clamp_min(eps)
    else:
        # Cosines are already non-negative: invalid/non-positive alignments
        # were set to zero above. Aggregate only real response tokens, then
        # reuse one normalized teacher mixture throughout each rollout.
        valid_response = response_mask.detach().to(device=cosine.device).bool()
        masked_cosine = cosine * valid_response.unsqueeze(0)
        token_count = valid_response.sum(dim=-1).clamp_min(1).to(cosine.dtype)
        rollout_scores = masked_cosine.sum(dim=-1) / token_count.unsqueeze(0)
        rollout_weights = rollout_scores / rollout_scores.sum(dim=0, keepdim=True).clamp_min(eps)
        weights = rollout_weights.unsqueeze(-1).expand_as(cosine)
        weights = weights * valid_response.unsqueeze(0)
    return weights, cosine, specialization_norm


def _compute_delta_advantage_metrics(
    advantages: torch.Tensor,
    router_weights: torch.Tensor,
    response_mask: torch.Tensor,
    zero_eps: float = 1e-8,
    near_zero_eps: float = 1e-4,
) -> dict[str, torch.Tensor]:
    """Measure how much usable signal remains after delta teacher routing.

    ``router_weights`` is teacher-major with shape ``[num_teachers, batch,
    response_length]``.  A zero final advantage can either come from every
    teacher being filtered or from cancellation among selected teachers; the
    returned metrics distinguish those cases.  Rollout-level metrics average
    over non-empty rollouts so long responses do not dominate the result.
    """
    if advantages.shape != response_mask.shape:
        raise ValueError(
            f"advantages and response_mask must have identical shapes, got "
            f"{advantages.shape} and {response_mask.shape}."
        )
    if router_weights.ndim != 3 or router_weights.shape[1:] != advantages.shape:
        raise ValueError(
            "router_weights must have shape [num_teachers, batch, response_length], "
            f"got {router_weights.shape} for advantages {advantages.shape}."
        )
    if not 0 <= zero_eps <= near_zero_eps:
        raise ValueError(
            f"Expected 0 <= zero_eps <= near_zero_eps, got {zero_eps} and {near_zero_eps}."
        )

    mask = response_mask.bool()
    valid_tokens = mask.sum().clamp_min(1)
    abs_advantage = advantages.detach().float().abs()
    selected = router_weights.detach().float().sum(dim=0) > 0
    is_zero = abs_advantage <= zero_eps
    is_near_zero = abs_advantage <= near_zero_eps

    token_metrics = {
        "final_adv_zero_ratio": (is_zero & mask).sum() / valid_tokens,
        "final_adv_near_zero_ratio": (is_near_zero & mask).sum() / valid_tokens,
        "cancellation_ratio": (selected & is_zero & mask).sum() / valid_tokens,
        "mean_abs_final_adv": (abs_advantage * mask).sum() / valid_tokens,
    }

    rollout_token_counts = mask.sum(dim=-1)
    nonempty_rollouts = rollout_token_counts > 0
    rollout_denominator = nonempty_rollouts.sum().clamp_min(1)
    rollout_all_zero = ((is_zero | ~mask).all(dim=-1)) & nonempty_rollouts
    rollout_effective_ratio = ((~is_near_zero & mask).sum(dim=-1).float()) / rollout_token_counts.clamp_min(1)
    token_metrics["rollout_all_zero_ratio"] = rollout_all_zero.sum() / rollout_denominator
    token_metrics["rollout_mean_effective_token_ratio"] = (
        rollout_effective_ratio * nonempty_rollouts
    ).sum() / rollout_denominator
    return token_metrics


def _resolve_domain_teacher_indices(
    opd_teacher,
    teacher_names: list[str],
    batch_size: int,
    device: torch.device,
    fallback_teacher_name: str | None = None,
) -> torch.Tensor:
    """Map non-tensor domain labels to teacher indices.

    Missing and unknown labels use ``fallback_teacher_name`` when it is
    configured. Otherwise they preserve the historical primary-teacher
    fallback. ``opd_teacher`` may be a list, tuple, NumPy array, or scalar.
    """
    fallback_index = 0
    if fallback_teacher_name is not None:
        fallback_index = next(
            (
                index
                for index, name in enumerate(teacher_names)
                if str(name).strip().lower().replace("-", "_")
                == fallback_teacher_name.strip().lower().replace("-", "_")
            ),
            len(teacher_names) - 1,
        )

    if opd_teacher is None:
        return torch.full((batch_size,), fallback_index, device=device, dtype=torch.long)
    if isinstance(opd_teacher, (str, bytes)):
        labels = [opd_teacher] * batch_size
    elif hasattr(opd_teacher, "tolist"):
        labels = opd_teacher.tolist()
    else:
        labels = list(opd_teacher)
    if not isinstance(labels, list):
        labels = [labels] * batch_size
    if len(labels) != batch_size:
        raise ValueError(f"Expected {batch_size} domain labels, got {len(labels)}.")

    teacher_to_index = {name: index for index, name in enumerate(teacher_names)}
    selected_indices = []
    for label in labels:
        if isinstance(label, bytes):
            label = label.decode("utf-8")
        selected_indices.append(teacher_to_index.get(str(label), fallback_index))

    return torch.tensor(selected_indices, device=device, dtype=torch.long)


def _domain_route_teacher_signal(
    teacher_log_probs: torch.Tensor,
    opd_teacher,
    teacher_names: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one teacher per sample from a token-aligned teacher pool.

    Unknown labels preserve the historical static domain-routing behavior by
    falling back to the primary teacher (index 0).
    """
    if teacher_log_probs.ndim != 3:
        raise ValueError(
            "Domain-label routing requires teacher_log_probs with shape "
            f"[num_teachers, batch, response_length], got {teacher_log_probs.shape}."
        )
    if len(teacher_names) != teacher_log_probs.shape[0]:
        raise ValueError(
            f"Configured {len(teacher_names)} domain teacher names for "
            f"{teacher_log_probs.shape[0]} teacher tensors."
        )

    selected = _resolve_domain_teacher_indices(
        opd_teacher,
        teacher_names,
        teacher_log_probs.shape[1],
        teacher_log_probs.device,
    )
    batch_first = teacher_log_probs.movedim(0, 1)
    gather_index = selected[:, None, None].expand(-1, 1, batch_first.shape[-1])
    routed_log_probs = batch_first.gather(dim=1, index=gather_index).squeeze(1)
    return routed_log_probs, selected


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        return_topk=False,
        topk=0,
    ):
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            topk_token_ids/topk_log_probs: optional ``(bs, response_len, topk)``
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            topk_token_ids = None
            topk_log_probs = None
            requested_topk_token_ids = micro_batch.get("student_topk_token_ids", None)
            need_topk = return_topk or requested_topk_token_ids is not None
            if need_topk and self.use_fused_kernels:
                raise NotImplementedError("delta routing requires use_fused_kernels=False to access vocabulary logits")
            if return_topk and requested_topk_token_ids is None and topk <= 1:
                raise ValueError(f"topk must be greater than one when generating top-k candidates, got {topk}.")
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                requested_topk_ids_rmpad = None
                if requested_topk_token_ids is not None:
                    requested_topk_token_ids = requested_topk_token_ids.to(input_ids.device)
                    topk_width = requested_topk_token_ids.shape[-1]
                    full_topk_ids = torch.zeros(
                        batch_size,
                        seqlen,
                        topk_width,
                        device=input_ids.device,
                        dtype=requested_topk_token_ids.dtype,
                    )
                    full_topk_ids[:, -response_length - 1 : -1] = requested_topk_token_ids
                    requested_topk_ids_rmpad = index_first_axis(
                        rearrange(full_topk_ids, "b s k -> (b s) k"), indices
                    )

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    if requested_topk_ids_rmpad is not None:
                        requested_topk_ids_rmpad, _, _ = ulysses_pad_and_slice_inputs(
                            requested_topk_ids_rmpad.transpose(0, 1),
                            position_ids_rmpad=None,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                        requested_topk_ids_rmpad = requested_topk_ids_rmpad.transpose(0, 1)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy or need_topk:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    if need_topk:
                        log_normalizer = torch.logsumexp(logits_rmpad.float(), dim=-1, keepdim=True)
                        if requested_topk_ids_rmpad is None:
                            topk_values_rmpad, topk_ids_rmpad = torch.topk(logits_rmpad.float(), k=topk, dim=-1)
                        else:
                            topk_ids_rmpad = requested_topk_ids_rmpad.long()
                            topk_values_rmpad = logits_rmpad.float().gather(dim=-1, index=topk_ids_rmpad)
                        topk_log_probs_rmpad = topk_values_rmpad - log_normalizer

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if need_topk:
                        topk_ids_rmpad = gather_outputs_and_unpad(
                            topk_ids_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        topk_log_probs_rmpad = gather_outputs_and_unpad(
                            topk_log_probs_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if need_topk:
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if need_topk:
                    topk_token_ids = full_topk_ids[:, -response_length - 1 : -1]
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if need_topk:
                        log_normalizer = torch.logsumexp(logits.float(), dim=-1, keepdim=True)
                        if requested_topk_token_ids is None:
                            topk_values, topk_token_ids = torch.topk(logits.float(), k=topk, dim=-1)
                        else:
                            topk_token_ids = requested_topk_token_ids.to(logits.device).long()
                            topk_values = logits.float().gather(dim=-1, index=topk_token_ids)
                        topk_log_probs = topk_values - log_normalizer
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if need_topk:
                return entropy, log_probs, topk_token_ids, topk_log_probs
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
        return_topk=False,
        topk=0,
        align_to_student_topk=True,
    ):
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if align_to_student_topk and "student_topk_token_ids" in data.batch.keys():
            select_keys.append("student_topk_token_ids")
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_token_ids_lst = []
        topk_log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                forward_output = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_topk=return_topk,
                    topk=topk,
                )
                if return_topk or "student_topk_token_ids" in model_inputs:
                    entropy, log_probs, topk_token_ids, topk_log_probs = forward_output
                else:
                    entropy, log_probs = forward_output
                    topk_token_ids = None
                    topk_log_probs = None
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)
            if topk_token_ids is not None:
                topk_token_ids_lst.append(topk_token_ids)
                topk_log_probs_lst.append(topk_log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        topk_token_ids = torch.concat(topk_token_ids_lst, dim=0) if topk_token_ids_lst else None
        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0) if topk_log_probs_lst else None

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if topk_token_ids is not None:
                topk_token_ids = restore_dynamic_batch(topk_token_ids, batch_idx_list)
                topk_log_probs = restore_dynamic_batch(topk_log_probs, batch_idx_list)

        if topk_token_ids is not None:
            return log_probs, entropys, topk_token_ids, topk_log_probs
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
         # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        if "base_log_prob" in data.batch.keys():
            select_keys.append("base_log_prob")
        if "base_ref_log_prob" in data.batch.keys():
            select_keys.append("base_ref_log_prob")
        if self.config.policy_loss.entropy_aware_router:
            aggregation = self.config.policy_loss.get("teacher_signal_aggregation", "delta")
            required_router_keys = {"teacher_log_probs"}
            if aggregation == "delta":
                required_router_keys.update(
                    {
                        "delta_router_weights",
                        "delta_router_alignment",
                        "delta_router_specialization_norm",
                    }
                )
            elif aggregation == "accessible_novelty":
                required_router_keys.add("teacher_accessible_novelty")
            else:
                required_router_keys.add("teacher_entropies")
            missing_router_keys = required_router_keys.difference(data.batch.keys())
            if missing_router_keys:
                raise ValueError(
                    "The configured token-level multi-teacher router is missing required "
                    f"signals {sorted(missing_router_keys)}. Configure at least two named "
                    "teacher models and the router-specific reference forward."
                )
            for key in required_router_keys:
                if key not in select_keys:
                    select_keys.append(key)
            if (
                aggregation == "entropy"
                and self.config.policy_loss.get("entropy_calibration_enabled", False)
            ):
                if "calibrated_teacher_entropies" not in data.batch.keys():
                    raise ValueError(
                        "entropy_calibration_enabled requires calibrated_teacher_entropies "
                        "prepared once per global rollout batch by the trainer"
                    )
                select_keys.append("calibrated_teacher_entropies")
        elif self.config.policy_loss.multi_teacher_distill and "teacher_log_probs" in data.batch.keys():
            select_keys.append("teacher_log_probs")
        # Include ref_log_prob for only_reverse_kl_advantages mode
        if self.config.policy_loss.only_reverse_kl_advantages and "ref_log_prob" in data.batch.keys():
            if "ref_log_prob" not in select_keys:
                select_keys.append("ref_log_prob")
        
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include opd_teacher for multi-teacher distillation
        if "opd_teacher" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("opd_teacher")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # only use reverse KL for advantages if only_reverse_kl_advantages is True
                    if self.config.policy_loss.only_reverse_kl_advantages:
                        lambda_vals = self.config.policy_loss.lambda_vals
                        if (
                            self.config.policy_loss.multi_teacher_distill
                            and self.config.policy_loss.entropy_aware_router
                        ):
                            teacher_log_probs = model_inputs["teacher_log_probs"].movedim(-1, 0)
                            aggregation = self.config.policy_loss.get("teacher_signal_aggregation", "delta")
                            if aggregation == "delta":
                                router_weights = model_inputs["delta_router_weights"].movedim(-1, 0).float()
                                if lambda_vals == 1.0:
                                    teacher_advantages = teacher_log_probs.float() - old_log_prob.float().unsqueeze(0)
                                else:
                                    if "base_log_prob" not in model_inputs:
                                        raise ValueError(
                                            "Delta-routed G-OPD (lambda_vals != 1) requires base_log_prob."
                                        )
                                    actor_base = model_inputs["base_log_prob"].float().unsqueeze(0)
                                    teacher_advantages = actor_base - old_log_prob.float().unsqueeze(0)
                                    teacher_advantages = teacher_advantages + (
                                        teacher_log_probs.float() - actor_base
                                    ) * lambda_vals
                                # Weights are computed on detached teacher/base/student top-k
                                # distributions.  Mixing the original per-teacher OPD
                                # advantages preserves the teacher-student update direction.
                                advantages = (router_weights * teacher_advantages).sum(dim=0)
                                reverse_kl = -advantages
                            else:
                                raw_teacher_entropies = model_inputs.get("teacher_entropies")
                                teacher_entropies = raw_teacher_entropies
                                if (
                                    aggregation == "entropy"
                                    and self.config.policy_loss.get("entropy_calibration_enabled", False)
                                ):
                                    teacher_entropies = model_inputs["calibrated_teacher_entropies"]
                                if teacher_entropies is not None:
                                    teacher_entropies = teacher_entropies.movedim(-1, 0)
                                if raw_teacher_entropies is not None:
                                    raw_teacher_entropies = raw_teacher_entropies.movedim(-1, 0)
                                teacher_utilities = model_inputs.get("teacher_accessible_novelty")
                                if teacher_utilities is not None:
                                    teacher_utilities = teacher_utilities.movedim(-1, 0)
                                routed_teacher_log_prob, router_weights = _entropy_route_teacher_signal(
                                    teacher_log_probs,
                                    teacher_entropies,
                                    temperature=self.config.policy_loss.entropy_router_temperature,
                                    aggregation=aggregation,
                                    teacher_utilities=teacher_utilities,
                                )
                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - routed_teacher_log_prob
                                else:
                                    if "base_log_prob" not in model_inputs:
                                        raise ValueError(
                                            "Entropy-routed G-OPD (lambda_vals != 1) requires base_log_prob."
                                        )
                                    reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                    reverse_kl = reverse_kl - (
                                        routed_teacher_log_prob - model_inputs["base_log_prob"]
                                    ) * lambda_vals

                            valid_tokens = response_mask.sum().clamp_min(1)
                            router_metric_name = {
                                "delta": "delta_router",
                                "accessible_novelty": "accessible_novelty",
                            }.get(aggregation, "entropy_router")
                            # Delta routing diagnostics are computed once on the full rollout
                            # batch in RayPPOTrainer. Emitting the same ratios from actor
                            # micro-batches would overwrite those exact step metrics after
                            # reduction. Entropy routing still relies on this actor-side path.
                            if aggregation != "delta":
                                for teacher_idx in range(router_weights.shape[0]):
                                    utilization = (router_weights[teacher_idx] * response_mask).sum() / valid_tokens
                                    micro_batch_metrics[
                                        f"actor/{router_metric_name}/teacher_{teacher_idx}_utilization"
                                    ] = utilization.detach().item() * loss_scale_factor
                            if aggregation == "delta":
                                advantage_metrics = _compute_delta_advantage_metrics(
                                    advantages,
                                    router_weights,
                                    response_mask,
                                )
                                for metric_name, metric_value in advantage_metrics.items():
                                    # Diagnostics are averaged by reduce_metrics; unlike the
                                    # loss, they must not be scaled for gradient accumulation.
                                    micro_batch_metrics[f"actor/delta_router/{metric_name}"] = (
                                        metric_value.detach().item()
                                    )
                            elif aggregation == "accessible_novelty":
                                utility_margin = (
                                    teacher_utilities.max(dim=0).values
                                    - teacher_utilities.min(dim=0).values
                                )
                                mean_margin = (utility_margin * response_mask).sum() / valid_tokens
                                micro_batch_metrics["actor/accessible_novelty/mean_utility_margin"] = (
                                    mean_margin.detach().item() * loss_scale_factor
                                )
                            else:
                                entropy_margin = (
                                    raw_teacher_entropies.max(dim=0).values
                                    - raw_teacher_entropies.min(dim=0).values
                                )
                                mean_margin = (entropy_margin * response_mask).sum() / valid_tokens
                                micro_batch_metrics["actor/entropy_router/mean_margin"] = (
                                    mean_margin.detach().item() * loss_scale_factor
                                )
                                calibrated_margin = (
                                    teacher_entropies.max(dim=0).values
                                    - teacher_entropies.min(dim=0).values
                                )
                                mean_calibrated_margin = (
                                    calibrated_margin * response_mask
                                ).sum() / valid_tokens
                                micro_batch_metrics["actor/entropy_router/calibrated_mean_margin"] = (
                                    mean_calibrated_margin.detach().item() * loss_scale_factor
                                )
                        elif (
                            self.config.policy_loss.multi_teacher_distill
                            and "teacher_log_probs" in model_inputs
                        ):
                            if "opd_teacher" not in model_inputs:
                                raise ValueError(
                                    "Domain-label multi-teacher OPD requires extra_info.opd_teacher "
                                    "for every training sample."
                                )
                            teacher_log_probs = model_inputs["teacher_log_probs"].movedim(-1, 0)
                            selected_teacher_log_prob, selected_teacher_indices = _domain_route_teacher_signal(
                                teacher_log_probs,
                                model_inputs["opd_teacher"],
                                list(self.config.policy_loss.domain_label_teacher_names),
                            )
                            if lambda_vals == 1.0:
                                reverse_kl = old_log_prob - selected_teacher_log_prob
                            else:
                                if "base_log_prob" not in model_inputs:
                                    raise ValueError(
                                        "Domain-label G-OPD (lambda_vals != 1) requires base_log_prob."
                                    )
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reverse_kl = reverse_kl - (
                                    selected_teacher_log_prob - model_inputs["base_log_prob"]
                                ) * lambda_vals

                            for teacher_index, teacher_name in enumerate(
                                self.config.policy_loss.domain_label_teacher_names
                            ):
                                utilization = (selected_teacher_indices == teacher_index).float().mean()
                                micro_batch_metrics[
                                    f"actor/domain_router/{teacher_name}_utilization"
                                ] = utilization.detach().item() * loss_scale_factor
                        # Corrected reverse KL with base model normalization if base log probs are available
                        # Formula: (log_prob_actor - log_prob_ref) - (log_prob_actor_base - log_prob_ref_base)
                        # This removes the base model bias from both actor and ref models
                        elif "base_log_prob" in model_inputs and "base_ref_log_prob" in model_inputs:
                            if self.config.policy_loss.multi_teacher_distill:
                                #### multi-teacher distillation ####
                                if "opd_teacher" in model_inputs:
                                    opd_teacher = model_inputs["opd_teacher"]
                                    batch_size = old_log_prob.shape[0]

                                    reverse_kl = torch.zeros_like(old_log_prob)

                                    for i in range(batch_size):
                                        teacher_type = opd_teacher[i] if isinstance(opd_teacher, (list, tuple)) else opd_teacher
                                        # TODO: need to improve the logic here
                                        if teacher_type == "math":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        elif teacher_type == "code":
                                            if lambda_vals == 1.0:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_ref_log_prob"][i]
                                            else:
                                                reverse_kl[i] = old_log_prob[i] - model_inputs["base_log_prob"][i] - (model_inputs["base_ref_log_prob"][i] - model_inputs["base_log_prob"][i]) * lambda_vals
                                        else:
                                            reverse_kl[i] = old_log_prob[i] - model_inputs["ref_log_prob"][i]
                                else:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                #### multi-teacher distillation ####
                            else:
                                #### single-teacher distillation ####
                                reverse_kl = old_log_prob - model_inputs["base_log_prob"]
                                reward_correction = model_inputs["ref_log_prob"] - model_inputs["base_log_prob"]

                                if lambda_vals == 1.0:
                                    reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                                else:
                                    reverse_kl = reverse_kl - reward_correction * lambda_vals
                                #### single-teacher distillation ####
                        else:
                            # Standard reverse KL: log(π_actor / π_ref) = log_prob_actor - log_prob_ref
                            reverse_kl = old_log_prob - model_inputs["ref_log_prob"]
                        advantages = (- (reverse_kl))
                   
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
