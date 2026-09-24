import pytest
import torch

from verl.workers.actor.dp_actor import (
    _compute_delta_advantage_metrics,
    _delta_route_teacher_signal,
    _domain_route_teacher_signal,
    _entropy_route_teacher_signal,
    _resolve_domain_teacher_indices,
)


def test_entropy_router_softmaxes_negative_entropy_per_token():
    teacher_log_probs = torch.tensor(
        [
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]],
        ]
    )
    teacher_entropies = torch.tensor(
        [
            [[0.1, 0.9, 0.2], [0.7, 0.2, 0.8]],
            [[0.6, 0.3, 0.4], [0.1, 0.5, 0.2]],
        ]
    )

    routed, weights = _entropy_route_teacher_signal(teacher_log_probs, teacher_entropies)

    expected_weights = torch.softmax(-teacher_entropies / 0.1, dim=0)
    expected = (expected_weights * teacher_log_probs).sum(dim=0)
    torch.testing.assert_close(routed, expected)
    torch.testing.assert_close(weights, expected_weights)


def test_soft_entropy_router_blends_teacher_signals():
    teacher_log_probs = torch.tensor([[[-1.0]], [[-3.0]]])
    teacher_entropies = torch.tensor([[[0.0]], [[1.0]]])

    routed, weights = _entropy_route_teacher_signal(
        teacher_log_probs, teacher_entropies, temperature=1.0
    )

    expected_weights = torch.softmax(torch.tensor([0.0, -1.0]), dim=0)
    torch.testing.assert_close(weights[:, 0, 0], expected_weights)
    torch.testing.assert_close(routed, (expected_weights * torch.tensor([-1.0, -3.0])).sum().reshape(1, 1))


def test_entropy_router_supports_more_than_two_teachers():
    teacher_log_probs = torch.tensor([[[-1.0, -2.0]], [[-3.0, -4.0]], [[-5.0, -6.0]]])
    teacher_entropies = torch.tensor([[[0.5, 0.4]], [[0.2, 0.8]], [[0.7, 0.1]]])

    routed, weights = _entropy_route_teacher_signal(teacher_log_probs, teacher_entropies)

    expected_weights = torch.softmax(-teacher_entropies / 0.1, dim=0)
    torch.testing.assert_close(routed, (expected_weights * teacher_log_probs).sum(dim=0))
    assert torch.equal(weights.argmax(dim=0), torch.tensor([[1, 2]]))


def test_mean_aggregation_averages_three_teacher_signals_per_token():
    teacher_log_probs = torch.tensor(
        [[[-1.0, -2.0]], [[-4.0, -5.0]], [[-7.0, -8.0]]]
    )
    # Deliberately make a different teacher the entropy winner at each token:
    # mean aggregation must ignore those rankings.
    teacher_entropies = torch.tensor(
        [[[0.1, 0.9]], [[0.5, 0.2]], [[0.8, 0.4]]]
    )

    routed, weights = _entropy_route_teacher_signal(
        teacher_log_probs,
        teacher_entropies,
        temperature=0.0,
        aggregation="mean",
    )

    torch.testing.assert_close(routed, teacher_log_probs.mean(dim=0))
    torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / 3.0))
    torch.testing.assert_close(weights.sum(dim=0), torch.ones_like(routed))
    old_log_probs = torch.tensor([[-0.5, -1.5]])
    torch.testing.assert_close(
        routed - old_log_probs,
        (teacher_log_probs - old_log_probs.unsqueeze(0)).mean(dim=0),
    )


def test_entropy_router_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="identical shapes"):
        _entropy_route_teacher_signal(torch.zeros(2, 1, 2), torch.zeros(2, 1, 3))

    with pytest.raises(ValueError, match="at least two teachers"):
        _entropy_route_teacher_signal(torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))

    with pytest.raises(ValueError, match="positive"):
        _entropy_route_teacher_signal(torch.zeros(2, 1, 2), torch.zeros(2, 1, 2), temperature=0.0)

    with pytest.raises(ValueError, match="teacher_signal_aggregation"):
        _entropy_route_teacher_signal(
            torch.zeros(2, 1, 2),
            torch.zeros(2, 1, 2),
            aggregation="median",
        )


def test_accessible_novelty_router_softmaxes_utility():
    teacher_log_probs = torch.tensor([[[-1.0, -2.0]], [[-3.0, -4.0]], [[-5.0, -6.0]]])
    teacher_utilities = torch.tensor([[[0.1, 0.4]], [[0.8, 0.2]], [[0.3, 0.9]]])

    routed, weights = _entropy_route_teacher_signal(
        teacher_log_probs,
        None,
        temperature=0.1,
        aggregation="accessible_novelty",
        teacher_utilities=teacher_utilities,
    )

    expected_weights = torch.softmax(teacher_utilities / 0.1, dim=0)
    torch.testing.assert_close(routed, (expected_weights * teacher_log_probs).sum(dim=0))
    torch.testing.assert_close(weights, expected_weights)
    assert torch.equal(weights.argmax(dim=0), teacher_utilities.argmax(dim=0))


def test_accessible_novelty_router_uses_mean_when_all_utilities_are_zero():
    teacher_log_probs = torch.tensor([[[-1.0]], [[-3.0]]])

    routed, weights = _entropy_route_teacher_signal(
        teacher_log_probs,
        None,
        aggregation="accessible_novelty",
        teacher_utilities=torch.zeros_like(teacher_log_probs),
    )

    torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
    torch.testing.assert_close(routed, torch.tensor([[-2.0]]))


def test_delta_router_filters_negative_direction_alignment():
    student = torch.tensor([[[0.0, 0.0]]])
    teacher_base = torch.tensor([[[2.0, -2.0]]])
    teachers = torch.tensor(
        [
            [[[1.0, -1.0]]],  # specialization and teaching directions conflict
            [[[3.0, -3.0]]],  # specialization and teaching directions agree
        ]
    )

    weights, cosine, norms = _delta_route_teacher_signal(student, teachers, teacher_base)

    torch.testing.assert_close(weights[:, 0, 0], torch.tensor([0.0, 1.0]))
    assert cosine[0, 0, 0] == 0
    assert cosine[1, 0, 0] > 0
    assert torch.all(norms[:, 0, 0] > 0)


def test_delta_router_supports_uniform_and_normalized_cosine_weighting():
    student = torch.tensor([[[0.0, 1.0]]])
    teacher_base = torch.tensor([[[0.0, 0.0]]])
    teachers = torch.tensor(
        [
            [[[1.0, 0.0]]],
            [[[1.0, 2.0]]],
        ]
    )

    uniform, cosine, _ = _delta_route_teacher_signal(
        student, teachers, teacher_base, weighting="uniform"
    )
    cosine_weights, _, _ = _delta_route_teacher_signal(
        student, teachers, teacher_base, weighting="cosine"
    )

    torch.testing.assert_close(uniform[:, 0, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(
        cosine_weights[:, 0, 0], cosine[:, 0, 0] / cosine[:, 0, 0].sum()
    )


def test_delta_router_domain_only_keeps_selected_teacher_after_gate():
    student = torch.tensor(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ]
    )
    teacher_base = torch.zeros_like(student)
    teachers = torch.tensor(
        [
            [[[1.0, 0.0]], [[1.0, 0.0]]],
            [[[1.0, 2.0]], [[1.0, 2.0]]],
            [[[0.0, 2.0]], [[0.0, 2.0]]],
        ]
    )

    weights, cosine, _ = _delta_route_teacher_signal(
        student,
        teachers,
        teacher_base,
        weighting="domain_only",
        domain_teacher_indices=torch.tensor([0, 2]),
    )

    expected = torch.zeros_like(weights)
    expected[0, 0] = 1.0
    expected[2, 1] = 1.0
    torch.testing.assert_close(weights, expected)


def test_delta_router_rollout_level_reuses_valid_token_mean_cosine_weights():
    student = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [0.0, 0.0]]])
    teacher_base = torch.zeros_like(student)
    teachers = torch.tensor(
        [
            [[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]],
        ]
    )
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    weights, cosine, _ = _delta_route_teacher_signal(
        student,
        teachers,
        teacher_base,
        weighting="rollout_level",
        response_mask=response_mask,
    )

    rollout_scores = cosine[:, 0, :2].mean(dim=-1)
    expected_rollout_weights = rollout_scores / rollout_scores.sum()
    torch.testing.assert_close(weights[:, 0, 0], expected_rollout_weights)
    torch.testing.assert_close(weights[:, 0, 1], expected_rollout_weights)
    torch.testing.assert_close(weights[:, 0, 2], torch.zeros(3))


def test_delta_router_rollout_level_requires_response_mask():
    with pytest.raises(ValueError, match="requires response_mask"):
        _delta_route_teacher_signal(
            torch.zeros(1, 1, 2),
            torch.zeros(2, 1, 1, 2),
            torch.zeros(1, 1, 2),
            weighting="rollout_level",
        )


def test_delta_router_rejects_unknown_weighting():
    with pytest.raises(ValueError, match="delta_router_weighting"):
        _delta_route_teacher_signal(
            torch.zeros(1, 1, 2),
            torch.zeros(2, 1, 1, 2),
            torch.zeros(1, 1, 2),
            weighting="winner_take_all",
        )


def test_delta_router_uses_raw_topk_vectors_without_centering():
    student = torch.tensor([[[0.0, 0.0]]])
    teacher_base = torch.tensor([[[0.0, 0.0]]])
    teachers = torch.tensor(
        [
            [[[1.0, 1.0]]],
            [[[2.0, 2.0]]],
        ]
    )

    weights, cosine, norms = _delta_route_teacher_signal(
        student, teachers, teacher_base, weighting="cosine"
    )

    expected_norms = torch.tensor([2.0**0.5, 8.0**0.5])
    torch.testing.assert_close(norms[:, 0, 0], expected_norms)
    torch.testing.assert_close(weights[:, 0, 0], torch.tensor([0.5, 0.5]))
    torch.testing.assert_close(cosine[:, 0, 0], torch.ones(2))


def test_delta_router_returns_zero_weights_when_no_teacher_is_valid():
    student = torch.tensor([[[0.0, 0.0]]])
    teacher_base = torch.tensor([[[0.0, 0.0]]])
    teachers = torch.zeros(2, 1, 1, 2)

    weights, cosine, norms = _delta_route_teacher_signal(student, teachers, teacher_base)

    torch.testing.assert_close(weights, torch.zeros_like(weights))
    torch.testing.assert_close(cosine, torch.zeros_like(cosine))
    torch.testing.assert_close(norms, torch.zeros_like(norms))
    assert torch.isfinite(weights).all()


def test_delta_router_requires_inner_product_above_alignment_epsilon():
    student = torch.tensor([[[0.0, 0.0]]])
    teacher_base = torch.tensor([[[0.0, 0.0]]])
    teachers = torch.tensor(
        [
            [[[5e-4, 5e-4]]],  # inner product = 5e-7: below the margin
            [[[1e-3, 1e-3]]],  # inner product = 2e-6: above the margin
        ]
    )

    weights, cosine, _ = _delta_route_teacher_signal(
        student,
        teachers,
        teacher_base,
        alignment_epsilon=1e-6,
    )

    torch.testing.assert_close(weights[:, 0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(cosine[:, 0, 0], torch.tensor([0.0, 1.0]))


def test_delta_advantage_metrics_separate_filtering_from_cancellation():
    advantages = torch.tensor(
        [
            [0.0, 0.0, 2e-5, 0.5],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    # Token 0 is all-filtered. Token 1 has a selected teacher but its final
    # advantage is exactly zero, so it is counted as cancellation.
    weights = torch.tensor(
        [
            [[0.0, 1.0, 0.5, 1.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 0, 0],
        ]
    )

    metrics = _compute_delta_advantage_metrics(advantages, weights, response_mask)

    assert metrics["final_adv_zero_ratio"] == pytest.approx(4 / 6)
    assert metrics["final_adv_near_zero_ratio"] == pytest.approx(5 / 6)
    assert metrics["cancellation_ratio"] == pytest.approx(1 / 6)
    assert metrics["mean_abs_final_adv"] == pytest.approx(0.50002 / 6)
    assert metrics["rollout_all_zero_ratio"] == pytest.approx(0.5)
    assert metrics["rollout_mean_effective_token_ratio"] == pytest.approx(0.125)


def test_domain_router_selects_math_code_and_instruction_following():
    teacher_log_probs = torch.tensor(
        [
            [[-1.0, -1.1], [-1.2, -1.3], [-1.4, -1.5]],
            [[-2.0, -2.1], [-2.2, -2.3], [-2.4, -2.5]],
            [[-3.0, -3.1], [-3.2, -3.3], [-3.4, -3.5]],
        ]
    )

    routed, selected = _domain_route_teacher_signal(
        teacher_log_probs,
        ["math", "code", "instruction_following"],
        ["math", "code", "instruction_following"],
    )

    torch.testing.assert_close(
        routed,
        torch.tensor([[-1.0, -1.1], [-2.2, -2.3], [-3.4, -3.5]]),
    )
    assert torch.equal(selected, torch.tensor([0, 1, 2]))


def test_domain_router_unknown_label_falls_back_to_primary_teacher():
    teacher_log_probs = torch.tensor([[[-1.0]], [[-2.0]], [[-3.0]]])

    routed, selected = _domain_route_teacher_signal(
        teacher_log_probs,
        ["unknown"],
        ["math", "code", "instruction_following"],
    )

    torch.testing.assert_close(routed, torch.tensor([[-1.0]]))
    assert torch.equal(selected, torch.tensor([0]))


def test_domain_indices_fall_back_to_instruction_following_for_domain_routing():
    teacher_names = ["math", "code", "instruction_following"]

    missing = _resolve_domain_teacher_indices(
        None,
        teacher_names,
        batch_size=2,
        device=torch.device("cpu"),
        fallback_teacher_name="instruction_following",
    )
    unknown = _resolve_domain_teacher_indices(
        ["math", "unknown", "code"],
        teacher_names,
        batch_size=3,
        device=torch.device("cpu"),
        fallback_teacher_name="instruction_following",
    )

    assert torch.equal(missing, torch.tensor([2, 2]))
    assert torch.equal(unknown, torch.tensor([0, 2, 1]))
