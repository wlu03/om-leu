"""Head/slot alignment regularizer + diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_alignment_is_zero_for_diagonal_and_positive_for_collapsed_heads() -> None:
    from src.train.regularizers import head_slot_alignment

    B, J, K = 4, 3, 5
    eye = torch.eye(K).view(1, 1, K, K).expand(B, J, K, K) * 30.0  # head m loves slot m
    assert head_slot_alignment(eye).item() < 1e-6
    collapsed = torch.zeros(B, J, K, K)
    collapsed[:, :, 0, :] = 30.0  # every head loves slot 0
    loss = head_slot_alignment(collapsed)
    assert loss.item() > 10.0
    # Gradient flows.
    A = torch.randn(B, J, K, K, requires_grad=True)
    head_slot_alignment(A).backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()


def test_alignment_noop_when_K_differs_from_M() -> None:
    from src.train.regularizers import head_slot_alignment

    A = torch.randn(2, 3, 3, 5, requires_grad=True)  # K=3, M=5
    loss = head_slot_alignment(A)
    assert loss.item() == 0.0
    loss.backward()  # still differentiable


def test_diagnostics_shapes_and_agreement() -> None:
    from src.train.regularizers import head_slot_alignment_diagnostics

    B, J, K = 3, 2, 5
    eye = torch.eye(K).view(1, 1, K, K).expand(B, J, K, K) + 0.01 * torch.randn(B, J, K, K)
    d = head_slot_alignment_diagnostics(eye)
    assert d["slot_argmax_agreement"] == pytest.approx(1.0)
    assert len(d["mean_score_matrix"]) == K and len(d["mean_score_matrix"][0]) == K
    assert len(d["head_score_correlation"]) == K
    assert d["per_head_top_slot"] == list(range(K))
    collapsed = torch.randn(B, J, K, 1).expand(B, J, K, K).clone()  # identical heads
    dc = head_slot_alignment_diagnostics(collapsed)
    assert all(abs(v - 1.0) < 1e-6 for row in dc["head_score_correlation"] for v in row)
    assert dc["slot_argmax_agreement"] <= 0.5


def test_config_reads_head_alignment_from_given_yaml() -> None:
    from src.train.regularizers import RegularizerConfig

    default = RegularizerConfig.from_default()
    assert default.head_alignment == 0.0
    aligned = RegularizerConfig.from_default(REPO_ROOT / "configs" / "head_aligned.yaml")
    assert aligned.head_alignment == pytest.approx(0.1)
    # Everything else identical to the default config.
    for f in ("weight_l2", "salience_entropy", "diversity", "head_variance",
              "monotonicity", "monotonicity_enabled"):
        assert getattr(aligned, f) == getattr(default, f)


def test_combined_regularizer_includes_alignment_term() -> None:
    from src.model.po_leu import POLEU
    from src.train.regularizers import RegularizerConfig, combined_regularizer

    torch.manual_seed(0)
    model = POLEU(K=5, J=3, d_e=16, p=4, M=5, attribute_hidden=8)
    z = torch.randn(2, 4)
    E = torch.nn.functional.normalize(torch.randn(2, 3, 5, 16), dim=-1)
    _, inter = model(z, E)
    off = RegularizerConfig(head_alignment=0.0, monotonicity_enabled=False)
    on = RegularizerConfig(head_alignment=0.5, monotonicity_enabled=False)
    r_off = combined_regularizer(model, inter, E, cfg=off)
    r_on = combined_regularizer(model, inter, E, cfg=on)
    assert r_on.item() > r_off.item()


def test_run_all_reports_writes_head_alignment(tmp_path: Path) -> None:
    from src.eval.interpret import run_all_reports
    from src.model.po_leu import POLEU

    torch.manual_seed(0)
    model = POLEU(K=5, J=3, d_e=16, p=4, M=5, attribute_hidden=8)
    z = torch.randn(2, 4)
    E = torch.nn.functional.normalize(torch.randn(2, 3, 5, 16), dim=-1)
    outcomes = [[[f"o{b}{j}{k}" for k in range(5)] for j in range(3)] for b in range(2)]
    bundle = run_all_reports(model, z, E, torch.zeros(2, dtype=torch.int64), outcomes,
                             out_dir=tmp_path, head_names=["a", "b", "c", "d", "e"])
    assert (tmp_path / "head_alignment.json").exists()
    assert bundle["head_alignment"]["head_names"] == ["a", "b", "c", "d", "e"]
    assert 0.0 <= bundle["head_alignment"]["slot_argmax_agreement"] <= 1.0
