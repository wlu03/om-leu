"""E5 synthetic tests.  These are fixtures, not empirical results."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from omleu_experiments.e5.risk import RiskSensitiveHead, data_support_audit, mean_variance_utility


def test_covariance_is_positive_semidefinite():
    torch.manual_seed(0)
    head = RiskSensitiveHead(r=8, K=5)
    H = torch.randn(6, 3, 5, 8)
    S = head.sigma(H)
    assert S.shape == (6, 3, 5, 5)
    assert torch.allclose(S, S.transpose(-1, -2), atol=1e-6)
    ev = torch.linalg.eigvalsh(S)
    assert float(ev.min()) >= -1e-6


def test_zero_risk_aversion_reduces_to_the_expected_outcome_model():
    torch.manual_seed(0)
    w = torch.softmax(torch.randn(4, 5), -1)
    mu = torch.randn(4, 3, 5); S = torch.randn(4, 3, 5, 2) @ torch.randn(4, 3, 2, 5)
    S = S @ S.transpose(-1, -2)
    u0 = mean_variance_utility(w, mu, S, torch.zeros(4))
    assert torch.allclose(u0, (mu * w[:, None, :]).sum(-1))


def test_more_variance_at_fixed_mean_lowers_utility_when_rho_positive():
    w = torch.softmax(torch.randn(4, 5), -1)
    mu = torch.randn(4, 3, 5)
    base = torch.eye(5).expand(4, 3, 5, 5).clone()
    u_small = mean_variance_utility(w, mu, base * 0.1, torch.full((4,), 0.5))
    u_large = mean_variance_utility(w, mu, base * 1.0, torch.full((4,), 0.5))
    assert (u_large <= u_small + 1e-6).all() and float((u_small - u_large).max()) > 0


def test_quadratic_term_is_nonnegative_for_psd_covariance():
    torch.manual_seed(1)
    head = RiskSensitiveHead(r=6, K=5)
    H = torch.randn(5, 2, 5, 6)
    S = head.sigma(H)
    w = torch.softmax(torch.randn(5, 5), -1)
    quad = torch.einsum("nk,njkl,nl->nj", w, S, w)
    assert float(quad.min()) >= -1e-6


def test_data_support_audit_blocks_all_three_datasets_with_a_reason():
    for ds in ("swissmetro", "optima", "lpmc"):
        a = data_support_audit(ds)
        assert a["supported"] is False and a["status"] == "blocked_data"
        assert a["would_enable"] and a["note"]
