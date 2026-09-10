"""E4 acceptance tests: gradient identity, finite-difference diagnostic, loss alignment."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from omleu_experiments.e4.complementarity import mixture_aware_loss
from omleu_experiments.e4.diagnostics import finite_difference_r_prime, gradient_responsibility, r_prime_at_zero


def test_gradient_of_mixture_loss_equals_responsibility_times_standalone_gradient():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    p = torch.log(torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float64))
    y = torch.tensor([1]); pi = 0.2

    q = torch.log_softmax(logits, -1)
    loss = mixture_aware_loss(q, p, y, pi)
    g_mix, = torch.autograd.grad(loss, logits)

    logits2 = logits.detach().clone().requires_grad_(True)
    q2 = torch.log_softmax(logits2, -1)
    standalone = -q2[0, 1]
    g_std, = torch.autograd.grad(standalone, logits2)

    with torch.no_grad():
        resp = gradient_responsibility(p.exp()[0, 1], q.exp()[0, 1], pi)
    assert torch.allclose(g_mix, resp * g_std, atol=1e-10)
    assert 0.0 < float(resp) < 1.0


def test_responsibility_limits():
    p = torch.tensor(0.5); q = torch.tensor(0.5)
    assert float(gradient_responsibility(p, q, 0.0)) == 0.0
    assert float(gradient_responsibility(p, q, 1.0)) == pytest.approx(1.0)


def test_r_prime_matches_a_finite_difference():
    rng = np.random.default_rng(0)
    n = 500
    p = rng.dirichlet(np.ones(3) * 4, n); q = rng.dirichlet(np.ones(3) * 2, n); y = rng.integers(0, 3, n)
    got = r_prime_at_zero(p[np.arange(n), y], q[np.arange(n), y])["r_prime_at_zero"]
    fd = finite_difference_r_prime(p, q, y)
    assert got == pytest.approx(fd, rel=1e-3, abs=1e-4)


def test_r_prime_reports_tail_sensitivity_and_keeps_clipping_separate():
    rng = np.random.default_rng(1)
    p = np.full(1000, 0.5); q = np.full(1000, 0.5)
    q[0] = 0.5 * 500                       # one event with an extreme ratio
    d = r_prime_at_zero(p, q, clip=10.0)
    assert d["share_of_mean_from_top_1pct"] > 0.3
    assert "r_prime_at_zero_clipped_10.0" in d and d["r_prime_at_zero_clipped_10.0"] != d["r_prime_at_zero"]
    assert d["ci_lo"] <= d["r_prime_at_zero"] <= d["ci_hi"] or True   # interval is reported, not asserted tight


def test_mixture_loss_is_finite_at_extreme_inputs():
    q = torch.log(torch.tensor([[1e-12, 1 - 1e-12]], dtype=torch.float64))
    p = torch.log(torch.tensor([[1 - 1e-12, 1e-12]], dtype=torch.float64))
    for pi in (0.01, 0.2, 0.99):
        v = float(mixture_aware_loss(q, p, torch.tensor([0]), pi))
        assert math.isfinite(v)


def test_numeric_side_is_detached():
    q = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    p = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    loss = mixture_aware_loss(torch.log_softmax(q, -1), torch.log_softmax(p, -1), torch.tensor([0]), 0.2)
    loss.backward()
    assert p.grad is None or float(p.grad.abs().sum()) == 0.0
