import pytest
import torch

from verl.utils.teacher_routing import accessible_novelty_from_topk, topk_log_probs_from_logits


def test_topk_log_probs_matches_full_log_softmax_across_chunks():
    torch.manual_seed(7)
    logits = torch.randn(2, 5, 11, dtype=torch.bfloat16)

    ids, log_probs = topk_log_probs_from_logits(logits, top_k=4, chunk_size=3)
    expected_log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected_values, expected_ids = expected_log_probs.topk(4, dim=-1)

    assert torch.equal(ids, expected_ids)
    torch.testing.assert_close(log_probs, expected_values)


def test_accessible_novelty_distinguishes_redundant_accessible_and_disjoint_teachers():
    student_ids = torch.tensor([[[1, 2, 3, 4]]])
    student_log_probs = torch.tensor([[[-0.8, -1.1, -1.5, -2.0]]])

    redundant = accessible_novelty_from_topk(
        student_ids, student_log_probs, student_ids, student_log_probs
    )
    accessible = accessible_novelty_from_topk(
        student_ids,
        student_log_probs,
        torch.tensor([[[1, 2, 3, 5]]]),
        torch.tensor([[[-1.8, -0.5, -1.2, -2.1]]]),
    )
    disjoint = accessible_novelty_from_topk(
        student_ids,
        student_log_probs,
        torch.tensor([[[5, 6, 7, 8]]]),
        torch.tensor([[[-0.5, -1.0, -1.5, -2.0]]]),
    )

    torch.testing.assert_close(redundant["utility"], torch.zeros(1, 1), atol=1e-7, rtol=0)
    assert accessible["utility"].item() > 0
    assert accessible["overlap_ratio"].item() == pytest.approx(0.75)
    torch.testing.assert_close(disjoint["utility"], torch.zeros(1, 1), atol=1e-7, rtol=0)


def test_novelty_only_removes_reachability_multiplier():
    student_ids = torch.tensor([[[1, 2, 3, 4]]])
    teacher_ids = torch.tensor([[[1, 2, 3, 5]]])
    student_log_probs = torch.tensor([[[-1.8, -2.1, -2.5, -3.0]]])
    teacher_log_probs = torch.tensor([[[-2.8, -1.5, -2.2, -3.1]]])

    default = accessible_novelty_from_topk(
        student_ids, student_log_probs, teacher_ids, teacher_log_probs
    )
    novelty_only = accessible_novelty_from_topk(
        student_ids,
        student_log_probs,
        teacher_ids,
        teacher_log_probs,
        utility_mode="novelty_only",
    )

    expected_reachability = torch.sqrt(
        novelty_only["student_overlap_mass"] * novelty_only["teacher_overlap_mass"]
    )
    torch.testing.assert_close(default["utility"], expected_reachability * novelty_only["utility"])
    assert novelty_only["utility"].item() > default["utility"].item()


def test_accessible_novelty_chunked_and_unchunked_match():
    torch.manual_seed(11)
    student_ids = torch.stack([torch.randperm(20)[:4] for _ in range(12)]).reshape(3, 4, 4)
    teacher_ids = torch.stack([torch.randperm(20)[:4] for _ in range(12)]).reshape(3, 4, 4)
    student_log_probs = torch.log_softmax(torch.randn(3, 4, 4), dim=-1)
    teacher_log_probs = torch.log_softmax(torch.randn(3, 4, 4), dim=-1)

    chunked = accessible_novelty_from_topk(
        student_ids,
        student_log_probs,
        teacher_ids,
        teacher_log_probs,
        token_chunk_size=3,
    )
    unchunked = accessible_novelty_from_topk(
        student_ids,
        student_log_probs,
        teacher_ids,
        teacher_log_probs,
        token_chunk_size=0,
    )

    for name in chunked:
        torch.testing.assert_close(chunked[name], unchunked[name])
