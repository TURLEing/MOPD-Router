import random
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _batch_calibrate_teacher_entropies


class _TokenizerStub:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def test_router_sample_index_minus_one_is_deterministic_random_without_global_rng_side_effects():
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 17
    trainer.config = OmegaConf.create({"data": {"seed": 123}})
    policy_config = OmegaConf.create({"entropy_router_log_sample_index": -1})

    random.seed(999)
    global_rng_state = random.getstate()
    selected = trainer._select_router_log_sample_index(policy_config, batch_size=8)

    expected = random.Random(123 * 1_000_003 + 17).randrange(8)
    assert selected == expected
    assert random.getstate() == global_rng_state


def test_router_sample_index_non_negative_wraps_by_batch_size():
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 0
    trainer.config = OmegaConf.create({"data": {"seed": 42}})

    assert trainer._select_router_log_sample_index(
        OmegaConf.create({"entropy_router_log_sample_index": 10}), batch_size=4
    ) == 2


def test_entropy_router_sample_logs_prompt_rollout_entropies_and_argmin_teacher(capsys):
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 10
    trainer.tokenizer = _TokenizerStub()
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "entropy_aware_router": True,
                        "entropy_router_log_sample": True,
                        "entropy_router_log_interval": 10,
                        "entropy_router_log_max_tokens": 2,
                        "entropy_router_log_sample_index": 0,
                        "teacher_signal_aggregation": "entropy",
                        "entropy_router_temperature": 0.1,
                    }
                },
                "ref": {
                    "model": {
                        "teacher_name": "math",
                        "base_model_path": "/models/code",
                        "base_model_teacher_name": "code",
                    }
                },
            }
        }
    )
    batch = SimpleNamespace(
        batch={
            "prompts": torch.tensor([[101, 102]]),
            "responses": torch.tensor([[201, 202, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]]),
            "teacher_entropies": torch.tensor([[[0.2, 0.8], [0.7, 0.1], [0.0, 0.0]]]),
        }
    )

    trainer._maybe_log_entropy_router_sample(batch)

    output = capsys.readouterr().out
    assert "prompt='101 102'" in output
    assert "displayed_student_rollout='201 202'" in output
    assert "'201' | math |" in output
    assert "math=0.2000, code=0.8000" in output
    assert "'202' | code |" in output
    assert "math=0.7000, code=0.1000" in output


def test_entropy_router_sample_respects_log_interval(capsys):
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 9
    trainer.tokenizer = _TokenizerStub()
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "entropy_aware_router": True,
                        "entropy_router_log_sample": True,
                        "entropy_router_log_interval": 10,
                    }
                }
            }
        }
    )

    trainer._maybe_log_entropy_router_sample(SimpleNamespace(batch={}))

    assert capsys.readouterr().out == ""


def test_entropy_router_step_metrics_average_valid_tokens_per_teacher():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "entropy_aware_router": True,
                        "teacher_signal_aggregation": "entropy",
                    }
                }
            }
        }
    )
    batch = SimpleNamespace(
        batch={
            "teacher_entropies": torch.tensor(
                [
                    [[0.2, 0.8], [0.4, 0.2], [99.0, 99.0]],
                    [[0.6, 0.5], [99.0, 99.0], [99.0, 99.0]],
                ]
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                ]
            ),
        }
    )

    metrics = trainer._compute_entropy_router_step_metrics(batch)

    assert metrics == {
        "actor/entropy_router/teacher_0_mean_entropy": pytest.approx(0.4),
        "actor/entropy_router/teacher_1_mean_entropy": pytest.approx(0.5),
    }


def test_delta_router_step_metrics_include_filtering_and_utilization():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "entropy_aware_router": True,
                        "teacher_signal_aggregation": "delta",
                    }
                }
            }
        }
    )
    batch = SimpleNamespace(
        batch={
            "delta_router_weights": torch.tensor(
                [[[0.25, 0.75], [0.0, 0.0], [9.0, 9.0]]]
            ),
            "delta_router_alignment": torch.tensor(
                [[[0.5, 0.2], [0.0, 0.0], [1.0, 1.0]]]
            ),
            "delta_router_specialization_norm": torch.tensor(
                [[[1.0, 3.0], [2.0, 4.0], [99.0, 99.0]]]
            ),
            "response_mask": torch.tensor([[1, 1, 0]]),
        }
    )

    metrics = trainer._compute_entropy_router_step_metrics(batch)

    assert metrics["actor/delta_router/selected_token_ratio"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/all_teachers_filtered_ratio"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/mean_positive_alignment_sum"] == pytest.approx(0.35)
    assert metrics["actor/delta_router/mean_effective_weight_mass"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/mean_max_positive_cosine"] == pytest.approx(0.25)
    assert metrics["actor/delta_router/mean_valid_teacher_count"] == pytest.approx(1.0)
    assert metrics["actor/delta_router/valid_teacher_count_0_ratio"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/valid_teacher_count_1_ratio"] == pytest.approx(0.0)
    assert metrics["actor/delta_router/valid_teacher_count_2_ratio"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/teacher_0_utilization"] == pytest.approx(0.125)
    assert metrics["actor/delta_router/teacher_1_utilization"] == pytest.approx(0.375)
    assert metrics["actor/delta_router/teacher_0_alignment_rate"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/teacher_1_alignment_rate"] == pytest.approx(0.5)
    assert metrics["actor/delta_router/teacher_0_mean_specialization_norm"] == pytest.approx(1.5)
    assert metrics["actor/delta_router/teacher_1_mean_specialization_norm"] == pytest.approx(3.5)


def test_entropy_calibration_normalizes_each_teacher_within_current_batch():
    entropies = torch.tensor(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [99.0, 99.0]],
        ]
    )
    mask = torch.tensor([[1, 1], [1, 0]])

    calibrated = _batch_calibrate_teacher_entropies(entropies, mask)
    valid = calibrated[mask.bool()]

    torch.testing.assert_close(valid.mean(dim=0), torch.zeros(2))
    torch.testing.assert_close(valid.square().mean(dim=0), torch.ones(2))


def test_accessible_novelty_step_metrics_average_valid_tokens():
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "policy_loss": {
                        "entropy_aware_router": True,
                        "teacher_signal_aggregation": "accessible_novelty",
                    }
                }
            }
        }
    )
    diagnostic_sums = torch.zeros(1, 2, 5)
    diagnostic_sums[0, :, 4] = torch.tensor([0.6, 1.4])
    batch = SimpleNamespace(
        batch={
            "accessible_novelty_diagnostic_sums": diagnostic_sums,
            "accessible_novelty_valid_token_counts": torch.tensor([2.0]),
        }
    )

    metrics = trainer._compute_entropy_router_step_metrics(batch)

    assert metrics["actor/accessible_novelty/teacher_0_utility"] == pytest.approx(0.3)
    assert metrics["actor/accessible_novelty/teacher_1_utility"] == pytest.approx(0.7)
