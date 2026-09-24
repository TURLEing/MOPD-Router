# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Distribution-level utilities shared by the multi-teacher routers."""

from __future__ import annotations

import math

import torch


def topk_log_probs_from_logits(
    logits: torch.Tensor,
    top_k: int,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact top-k token ids and log-probabilities in bounded chunks."""
    if logits.ndim < 2:
        raise ValueError(f"logits must have a vocabulary dimension, got {tuple(logits.shape)}")
    if top_k <= 0 or top_k > logits.shape[-1]:
        raise ValueError(f"top_k must be in [1, {logits.shape[-1]}], got {top_k}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    leading_shape = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    ids_chunks = []
    log_prob_chunks = []
    for chunk in flat_logits.split(chunk_size, dim=0):
        chunk_fp32 = chunk.float()
        top_values, top_ids = torch.topk(chunk_fp32, k=top_k, dim=-1)
        log_partition = torch.logsumexp(chunk_fp32, dim=-1, keepdim=True)
        ids_chunks.append(top_ids)
        log_prob_chunks.append(top_values - log_partition)

    top_ids = torch.cat(ids_chunks, dim=0).reshape(*leading_shape, top_k)
    top_log_probs = torch.cat(log_prob_chunks, dim=0).reshape(*leading_shape, top_k)
    return top_ids, top_log_probs


def accessible_novelty_from_topk(
    student_topk_ids: torch.Tensor,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    js_temperature: float = 0.1,
    utility_mode: str = "default",
    eps: float = 1e-8,
    token_chunk_size: int = 4096,
) -> dict[str, torch.Tensor]:
    """Score a teacher on the shared student/teacher top-k support.

    ``default`` uses
    ``sqrt(student_mass * teacher_mass) * (1 - exp(-JS / tau))``.
    ``novelty_only`` removes the overlap-mass multiplier.
    """
    shapes = {
        tuple(student_topk_ids.shape),
        tuple(student_topk_log_probs.shape),
        tuple(teacher_topk_ids.shape),
        tuple(teacher_topk_log_probs.shape),
    }
    if len(shapes) != 1:
        raise ValueError(f"all top-k tensors must have identical shapes, got {sorted(shapes)}")
    if student_topk_ids.ndim < 2:
        raise ValueError("top-k tensors must include token and top-k dimensions")
    if js_temperature <= 0 or not math.isfinite(js_temperature):
        raise ValueError(f"js_temperature must be finite and positive, got {js_temperature}")
    if utility_mode not in _ACCESSIBLE_NOVELTY_UTILITY_MODES:
        raise ValueError(
            "utility_mode must be one of "
            f"{sorted(_ACCESSIBLE_NOVELTY_UTILITY_MODES)}, got {utility_mode!r}"
        )
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    leading_shape = student_topk_ids.shape[:-1]
    num_tokens = math.prod(leading_shape)
    if token_chunk_size > 0 and num_tokens > token_chunk_size:
        flattened = [
            tensor.reshape(num_tokens, tensor.shape[-1])
            for tensor in (
                student_topk_ids,
                student_topk_log_probs,
                teacher_topk_ids,
                teacher_topk_log_probs,
            )
        ]
        chunks = {name: [] for name in _ACCESSIBLE_NOVELTY_KEYS}
        for start in range(0, num_tokens, token_chunk_size):
            stop = min(start + token_chunk_size, num_tokens)
            chunk_metrics = accessible_novelty_from_topk(
                *(tensor[start:stop] for tensor in flattened),
                js_temperature=js_temperature,
                utility_mode=utility_mode,
                eps=eps,
                token_chunk_size=0,
            )
            for name, value in chunk_metrics.items():
                chunks[name].append(value)
        return {
            name: torch.cat(values, dim=0).reshape(*leading_shape)
            for name, values in chunks.items()
        }

    matches = student_topk_ids.unsqueeze(-1).eq(teacher_topk_ids.unsqueeze(-2))
    student_in_overlap = matches.any(dim=-1)
    teacher_in_overlap = matches.any(dim=-2)

    student_probs = student_topk_log_probs.float().exp()
    teacher_probs = teacher_topk_log_probs.float().exp()
    student_overlap_probs = student_probs * student_in_overlap
    teacher_overlap_probs = teacher_probs * teacher_in_overlap

    student_mass = student_overlap_probs.sum(dim=-1)
    teacher_mass = teacher_overlap_probs.sum(dim=-1)
    overlap_count = student_in_overlap.sum(dim=-1)
    overlap_ratio = overlap_count.float() / student_topk_ids.shape[-1]

    teacher_on_student = (
        matches.to(teacher_probs.dtype) * teacher_probs.unsqueeze(-2)
    ).sum(dim=-1)
    student_normalized = student_overlap_probs / student_mass.unsqueeze(-1).clamp_min(eps)
    teacher_normalized = teacher_on_student / teacher_mass.unsqueeze(-1).clamp_min(eps)
    mixture = 0.5 * (student_normalized + teacher_normalized)

    student_kl = torch.where(
        student_in_overlap,
        student_normalized
        * (student_normalized.clamp_min(eps).log() - mixture.clamp_min(eps).log()),
        torch.zeros_like(student_normalized),
    ).sum(dim=-1)
    teacher_kl = torch.where(
        student_in_overlap,
        teacher_normalized
        * (teacher_normalized.clamp_min(eps).log() - mixture.clamp_min(eps).log()),
        torch.zeros_like(teacher_normalized),
    ).sum(dim=-1)
    overlap_js = 0.5 * (student_kl + teacher_kl)
    overlap_js = torch.where(overlap_count > 0, overlap_js, torch.zeros_like(overlap_js))

    reachability = torch.sqrt((student_mass * teacher_mass).clamp_min(0.0))
    saturated_novelty = -torch.expm1(-overlap_js / js_temperature)
    utility = reachability * saturated_novelty if utility_mode == "default" else saturated_novelty

    return {
        "overlap_ratio": overlap_ratio,
        "student_overlap_mass": student_mass,
        "teacher_overlap_mass": teacher_mass,
        "overlap_js": overlap_js,
        "utility": utility,
    }


_ACCESSIBLE_NOVELTY_KEYS = (
    "overlap_ratio",
    "student_overlap_mass",
    "teacher_overlap_mass",
    "overlap_js",
    "utility",
)

_ACCESSIBLE_NOVELTY_UTILITY_MODES = frozenset({"default", "novelty_only"})
