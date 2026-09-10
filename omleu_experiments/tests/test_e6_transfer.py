"""E6 acceptance tests: nesting of learning-curve subsets, frozen parameters, holdout support."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.harness.data import load_bundle
from omleu_experiments.e6.holdouts import compositional_holdout
from omleu_experiments.e6.transfer import SCHEMA_MAP, TransferReader
from omleu_experiments.runner import cluster_ids, person_strings, subsample_clusters


def test_learning_curve_subsets_are_nested_and_whole_respondents():
    clusters = np.array([f"p{i // 4}" for i in range(400)])
    rows = np.arange(400)
    sets = {f: set(subsample_clusters(rows, clusters, f, seed=7).tolist()) for f in (0.1, 0.25, 0.5, 1.0)}
    assert sets[0.1] <= sets[0.25] <= sets[0.5] <= sets[1.0]
    for f, s in sets.items():
        kept = {clusters[i] for i in s}
        for c in kept:                                  # every event of a kept respondent is kept
            assert {i for i in rows if clusters[i] == c} <= s


def test_transfer_freezes_the_scorer_and_trains_only_the_valuation():
    b = load_bundle("optima", 7)
    r = TransferReader(source="lpmc", axis_kw={"sensitivity": "global"}, freeze=True)
    r.n_members = 1
    m = r._member(b)
    for name, p in m.named_parameters():
        if name.startswith(("proj.", "norm.", "heads.")):
            p.requires_grad_(False)
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert any(n.startswith("val.") for n in trainable)
    assert all(n.startswith(("proj.", "norm.", "heads.")) for n in frozen) and frozen


def test_schema_map_is_explicit_about_missing_quantities():
    for axis, per_ds in SCHEMA_MAP.items():
        assert set(per_ds) == {"swissmetro", "optima", "lpmc"}
        for ds, (col, unit) in per_ds.items():
            assert (col is None) == (unit is None), f"{axis}/{ds} half-specified"


def test_compositional_holdout_refuses_thin_cells_and_reports_disjointness():
    b = load_bundle("lpmc", 7)
    persons, _, _ = person_strings(b)
    cl = cluster_ids(b, persons)
    dev = np.arange(b.N)
    out = compositional_holdout(b, dev, cl, feature_a="purpose=1", feature_b="interchanges", min_support=10**9)
    assert out["status"] == "blocked_data"
    out = compositional_holdout(b, dev, cl, feature_a="not_a_feature", feature_b="interchanges")
    assert out["status"] == "blocked_data" and "not a covariate" in out["reason"]
