"""Acceptance tests for the shared foundation (protocol, calibration, contracts, controls)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from omleu_experiments.calibrate import apply_calibration, fit_mixture, mixture_logprob
from omleu_experiments.contracts import Event, GenerationInput, generation_input, stable_hash
from omleu_experiments.metrics import all_metrics, cluster_draws, ece_top_label, paired_diff
from omleu_experiments.protocol import ordering_support, person_disjoint, rolling_origin
from omleu_experiments.views import inner_split


def _events(n=60, per=3):
    persons = [f"p{i // per}" for i in range(n)]
    return persons, [f"h{i // (per * 2)}" for i in range(n)]


def test_person_disjoint_has_no_cluster_overlap():
    persons, households = _events()
    part = person_disjoint(persons, households, master_seed=7, n_folds=5)
    tr = set(np.asarray(households)[part.train_dev]); te = set(np.asarray(households)[part.test])
    assert not (tr & te)
    seen = set()
    for f in part.dev_folds:
        cl = set(np.asarray(households)[f])
        assert not (cl & seen), "a household appears in two development folds"
        seen |= cl


def test_inner_split_is_cluster_disjoint():
    persons, households = _events()
    rows = np.arange(len(persons))
    f, v = inner_split(rows, np.asarray(households), seed=3)
    assert not (set(np.asarray(households)[f]) & set(np.asarray(households)[v]))


def test_temporal_blocks_are_forward_only():
    t = np.arange(200).astype(float)
    part = rolling_origin(t, [f"p{i}" for i in range(200)], n_folds=4)
    assert t[part.test].min() > t[part.train_dev].max()
    prev = -np.inf
    for f in part.dev_folds:
        assert t[f].min() > prev
        prev = t[f].max()


def test_ordering_support_rejects_task_index_and_synthetic():
    assert not ordering_support([1, 2, 3] * 50, "task_index", ["a"] * 150)["supported"]
    assert not ordering_support(list(range(4)) * 30, "synthetic", ["a"] * 120)["supported"]
    assert ordering_support(list(range(500)), "calendar", [f"p{i}" for i in range(500)])["supported"]


def test_generation_input_rejects_labels_and_person_label_aggregates():
    ev = Event(dataset="d", event_id="e", person_id="p", alt_ids=("a", "b"), avail=(True, True),
               attrs={"time": (1.0, 2.0)}, covariates={"age": 1.0}, history={"is_repeat": (0.0, 0.0)},
               chosen_index=0)
    ok = generation_input(ev, 0, "time", {"time": 1.0}, ["time"], "strict_prefix_own_events")
    assert ok.input_hash
    with pytest.raises(ValueError):
        generation_input(ev, 0, "time", {"time": 1.0, "chosen_index": 0}, ["time", "chosen_index"], "x")
    with pytest.raises(ValueError):
        generation_input(ev, 0, "time", {"novelty_rate": 0.3}, ["novelty_rate"], "x")


def test_event_requires_available_chosen_alternative():
    with pytest.raises(AssertionError):
        Event(dataset="d", event_id="e", person_id="p", alt_ids=("a", "b"), avail=(False, True),
              attrs={}, covariates={}, history={}, chosen_index=0)


def test_temperature_applied_exactly_once():
    u = torch.randn(30, 3, dtype=torch.float64)
    g = torch.tensor(-40.0, dtype=torch.float64); la = torch.tensor(0.3, dtype=torch.float64)
    got = mixture_logprob(u, None, g, la)
    want = torch.log_softmax(u * torch.exp(la), -1)
    assert torch.allclose(got, want)
    twice = torch.log_softmax(u * torch.exp(la) * torch.exp(la), -1)
    assert not torch.allclose(got, twice)


def test_mixture_boundaries_and_masking():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(50, 3)); q = np.log(rng.dirichlet(np.ones(3), 50)); y = rng.integers(0, 3, 50)
    cal = fit_mixture(u, q, y)
    assert 0.0 <= cal["pi"] <= 1.0 and cal["a"] > 0
    assert np.isfinite(cal["dev_nll_pi0"]) and np.isfinite(cal["dev_nll_pi1"])
    assert cal["grid_minus_lbfgs"] >= -1e-6, "L-BFGS is worse than a coarse grid"
    avail = np.ones((50, 3), dtype=bool); avail[:, 2] = False
    lp = apply_calibration(u, q, cal, avail=avail)
    p = np.exp(lp)
    assert np.allclose(p.sum(1), 1.0, atol=1e-9)


def test_paired_diff_uses_shared_clusters_and_sign_convention():
    base = np.array([1.0, 1.0, 1.0, 1.0]); var = np.array([0.5, 0.5, 0.5, 0.5])
    cl = ["a", "a", "b", "b"]
    d = paired_diff(base, var, cl, cluster_draws(cl, draws=200))
    assert d["mean"] == pytest.approx(0.5)      # positive = variant improves
    assert d["lo"] == pytest.approx(0.5) and d["n_clusters"] == 2


def test_ece_bins_are_fixed_and_bounded():
    lp = np.log(np.array([[0.9, 0.1], [0.6, 0.4]])); y = np.array([0, 1])
    e = ece_top_label(lp, y)
    assert 0.0 <= e <= 1.0
    assert all_metrics(lp, y)["n_events"] == 2
