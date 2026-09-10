"""E7 acceptance tests: gate inputs, shrinkage, and the monotonicity decomposition."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.harness.data import load_bundle
from omleu_experiments.calibrate import apply_conditional_gate, fit_conditional_gate, fit_mixture
from omleu_experiments.e7.probes import gate_monotonicity_probe
from omleu_experiments.runner import gate_context


def test_gate_context_contains_no_label_or_channel_correctness():
    b = load_bundle("optima", 7)
    rows = np.arange(64)
    C = gate_context(b, rows)
    assert C.shape == (64, 3) and np.isfinite(C).all()
    y = b.y[rows].numpy()
    for k in range(C.shape[1]):
        col = C[:, k]
        if col.std() < 1e-9:
            continue
        r = abs(np.corrcoef(col, y)[0, 1])
        assert r < 0.6, "a gate context feature tracks the label too closely to be admissible"


def test_strong_shrinkage_returns_the_global_gate():
    rng = np.random.default_rng(0); n = 400
    u = rng.normal(size=(n, 3)); q = np.log(rng.dirichlet(np.ones(3) * 3, n)); y = rng.integers(0, 3, n)
    C = rng.normal(size=(n, 3))
    strong = fit_conditional_gate(u, q, y, C, l2=1e6)
    assert max(abs(x) for x in strong["beta"]) < 1e-3
    glob = fit_mixture(u, q, y)
    assert strong["pi_mean"] == pytest.approx(glob["pi"], abs=0.05)


def test_gate_predictions_are_probabilities():
    rng = np.random.default_rng(1); n = 200
    u = rng.normal(size=(n, 3)); q = np.log(rng.dirichlet(np.ones(3), n)); y = rng.integers(0, 3, n)
    C = rng.normal(size=(n, 2))
    cal = fit_conditional_gate(u, q, y, C, l2=0.5)
    p = np.exp(apply_conditional_gate(u, q, C, cal))
    assert np.allclose(p.sum(1), 1.0, atol=1e-9) and (p >= 0).all()
    assert 0.0 <= cal["pi_mean"] <= 1.0


def test_monotonicity_decomposition_is_exact_and_names_the_gate_term():
    rng = np.random.default_rng(2); n = 50
    q_lo = rng.dirichlet(np.ones(3), n); q_hi = rng.dirichlet(np.ones(3), n)
    r_lo = rng.dirichlet(np.ones(3), n); r_hi = rng.dirichlet(np.ones(3), n)
    pi_lo = rng.uniform(0.1, 0.4, n); pi_hi = pi_lo + 0.05
    p_lo = (1 - pi_lo)[:, None] * q_lo + pi_lo[:, None] * r_lo
    p_hi = (1 - pi_hi)[:, None] * q_hi + pi_hi[:, None] * r_hi
    d = gate_monotonicity_probe(p_lo, p_hi, q_lo, q_hi, r_lo, r_hi, pi_lo, pi_hi, slot=0)
    assert d["decomposition_residual"] < 1e-12
    assert abs(d["term_gate"]) > 0.0
