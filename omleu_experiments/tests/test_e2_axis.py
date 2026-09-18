"""E2 acceptance tests: locality, equivariance, masking, axis integrity, weights."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.harness.data import load_bundle
from omleu_experiments.e2.axis_reader import AxisNet


@pytest.fixture(scope="module")
def bundle():
    return load_bundle("optima", 7)


def test_scores_are_local_to_their_axis_slot(bundle):
    m = AxisNet(bundle).eval()
    idx = torch.arange(4)
    c0 = m.scores(bundle, idx)
    b2 = bundle
    E = b2.E.clone(); E[idx, :, 2, :] = torch.randn_like(E[idx, :, 2, :])
    import dataclasses
    c1 = m.scores(dataclasses.replace(b2, E=E), idx)
    changed = (c1 - c0).abs().amax(dim=(0, 1))
    assert changed[2] > 1e-6, "the perturbed axis score did not move"
    others = [k for k in range(bundle.K) if k != 2]
    assert float(changed[others].max()) < 1e-6, "a head read a slot that is not its own axis"


def test_utility_changes_only_through_the_weighted_contribution(bundle):
    m = AxisNet(bundle, sensitivity="global").eval()
    idx = torch.arange(8)
    c = m.scores(bundle, idx)
    w = m.weights(bundle, idx); s = m.sensitivity_of(bundle, idx)
    u0 = m.utilities(bundle, idx, scores=c)
    delta = torch.zeros_like(c); delta[:, :, 1] = 0.25
    u1 = m.utilities(bundle, idx, scores=c + delta)
    expected = s[:, None] * w[:, 1][:, None] * 0.25
    assert torch.allclose(u1 - u0, expected, atol=1e-5)


def test_weights_are_a_simplex_and_sensitivity_is_bounded(bundle):
    for sens in ("fixed", "global", "person"):
        m = AxisNet(bundle, sensitivity=sens).eval()
        idx = torch.arange(16)
        w = m.weights(bundle, idx); s = m.sensitivity_of(bundle, idx)
        assert torch.allclose(w.sum(-1), torch.ones(len(idx)), atol=1e-6)
        assert (w >= 0).all() and (s >= 0).all() and (s <= m.s_max + 1e-6).all()


def test_bounded_change_implies_bounded_log_odds_change(bundle):
    """Elementary check of the diagnostic used in E3: with weights on a simplex,
    s <= s_max and every axis score moving by at most eps, the utility moves by at most
    s_max * eps and a pairwise log-odds by at most 2 * s_max * eps."""
    m = AxisNet(bundle, sensitivity="global").eval()
    idx = torch.arange(32)
    c = m.scores(bundle, idx); eps = 0.1
    pert = (torch.rand_like(c) * 2 - 1) * eps
    u0, u1 = m.utilities(bundle, idx, scores=c), m.utilities(bundle, idx, scores=c + pert)
    s_max = float(m.sensitivity_of(bundle, idx).max())
    assert float((u1 - u0).abs().max()) <= s_max * eps + 1e-5
    lo0 = u0[:, 0] - u0[:, 1]; lo1 = u1[:, 0] - u1[:, 1]
    assert float((lo1 - lo0).abs().max()) <= 2 * s_max * eps + 1e-5


def test_alternative_permutation_equivariance(bundle):
    import dataclasses
    m = AxisNet(bundle).eval()
    idx = torch.arange(6)
    lp = m(bundle, idx)
    perm = torch.tensor([2, 0, 1])
    b2 = dataclasses.replace(bundle, E=bundle.E[:, perm], Xnum=bundle.Xnum[:, perm],
                             Xhist=bundle.Xhist[:, perm], alt_idx=bundle.alt_idx[:, perm])
    lp2 = m(b2, idx)
    assert torch.allclose(lp[:, perm], lp2, atol=1e-6)


def test_masked_probabilities_sum_to_one_and_zero_the_unavailable(bundle):
    m = AxisNet(bundle).eval()
    idx = torch.arange(5)
    avail = torch.ones(len(idx), bundle.J, dtype=torch.bool); avail[:, 1] = False
    p = m(bundle, idx, avail=avail).exp()
    assert torch.allclose(p.sum(-1), torch.ones(len(idx)), atol=1e-6)
    assert float(p[:, 1].max()) < 1e-12


def test_wrong_axis_assignment_is_a_different_model(bundle):
    """Serialisation order does not matter while axis identity is preserved, but
    deliberately assigning text to the wrong axis changes the scores."""
    import dataclasses
    m = AxisNet(bundle).eval()
    idx = torch.arange(4)
    c = m.scores(bundle, idx)
    roll = dataclasses.replace(bundle, E=bundle.E.roll(1, dims=2))
    c_roll = m.scores(roll, idx)
    assert not torch.allclose(c, c_roll)
