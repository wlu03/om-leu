"""E3 acceptance tests: edits, paraphrases, losses, collapse diagnostics."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.harness.data import load_bundle
from omleu_experiments.e3.edits import EDITS, apply_edit
from omleu_experiments.e3.losses import collapse_diagnostics, consistency_losses, local_loss, order_loss, paraphrase_loss
from omleu_experiments.e3.paraphrase import paraphrase_text, preserves_claims, quantities
from omleu_experiments.sentences import template_sentences


class _Stub:
    """Minimal model exposing the outcome-score interface."""

    def __init__(self, s):
        self.s = s

    def scores(self, b, rows):
        return self.s


@pytest.fixture(scope="module")
def bundle():
    return load_bundle("optima", 7)


def test_edits_change_only_their_own_feature_and_linked_fields(bundle):
    spec = EDITS["fare_up"]
    X, elig = apply_edit(bundle, spec, slot=0)
    f = spec.feature_index(bundle)
    d = (X - bundle.Xnum)
    assert elig.any()
    assert torch.allclose(d[elig][:, 0, f], torch.full((int(elig.sum()),), spec.delta))
    other = [i for i in range(bundle.F) if i != f]
    assert float(d[:, :, other].abs().max()) == 0.0
    assert float(d[:, 1:, :].abs().max()) == 0.0, "an edit touched another alternative"


def test_linked_fields_stay_coherent(bundle):
    spec = EDITS["wait_up"]
    if spec.feature_index(bundle) is None:
        pytest.skip("dataset has no waiting attribute")
    X, elig = apply_edit(bundle, spec, slot=1)
    names = bundle.meta["alt_feature_names"]
    if "time_h" in names:
        d = (X - bundle.Xnum)[elig][:, 1, names.index("time_h")]
        assert torch.allclose(d, torch.full_like(d, 0.25)), "waiting rose without total time rising"


def test_edited_events_carry_no_label(bundle):
    spec = EDITS["fare_up"]
    X, _ = apply_edit(bundle, spec, slot=0)
    assert X.shape == bundle.Xnum.shape          # attributes only; y is untouched and unused


def test_paraphrases_preserve_quantities_and_negation(bundle):
    texts = [v for i in (0, 5, 17) for v in template_sentences(bundle, i, 0).values()]
    for t in texts:
        for fam in ("recorded_to_measured", "colon_to_clause", "passive", "notrecorded"):
            p = paraphrase_text(t, fam)
            assert preserves_claims(t, p), f"{fam} changed a claim: {t!r} -> {p!r}"
            assert quantities(t) == quantities(p)


def test_order_loss_only_penalises_improvement():
    s0 = torch.zeros(4, 3, 5)
    worse = s0.clone(); worse[:, 0, 0] = -1.0
    better = s0.clone(); better[:, 0, 0] = +1.0
    assert float(order_loss(s0, worse, [0], 0)) == 0.0
    assert float(order_loss(s0, better, [0], 0)) == pytest.approx(1.0)


def test_local_loss_penalises_spillover_only_on_declared_axes():
    s0 = torch.zeros(4, 3, 5)
    s1 = s0.clone(); s1[:, 0, 2] = 0.5
    assert float(local_loss(s0, s1, [1, 3], 0)) == 0.0
    assert float(local_loss(s0, s1, [2], 0)) == pytest.approx(0.25)


def test_consistency_losses_compose_and_are_optional():
    s = torch.randn(4, 3, 5)
    m = _Stub(s)
    out = consistency_losses(m, None, None, None, None, slot=0, affected=[0], unaffected=[1])
    assert out == {}
    sentinel = object()
    out = consistency_losses(m, sentinel, None, sentinel, None, slot=0, affected=[0], unaffected=[1], lam_para=1.0)
    assert "para" in out and float(out["para"]) == 0.0     # identical scores -> zero drift
    out = consistency_losses(m, sentinel, sentinel, None, None, slot=0, affected=[0], unaffected=[1],
                             lam_order=1.0, lam_local=1.0)
    assert set(out) == {"order", "local"}


def test_collapse_diagnostics_flag_a_constant_representation():
    const = torch.ones(10, 3, 5)
    d = collapse_diagnostics(const)
    assert max(d.values()) == 0.0
    varied = torch.randn(10, 3, 5)
    assert min(collapse_diagnostics(varied).values()) > 0.0
