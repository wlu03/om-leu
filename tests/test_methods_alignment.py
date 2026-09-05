"""methods/common/data.py must align raw numeric rows to the OM-LEU records."""
from pathlib import Path

import numpy as np
import pytest

from methods.common.data import ALTS, DEFAULT_RUN_TAGS, REPO_ROOT, load_dataset, records_path


@pytest.mark.parametrize("dataset", sorted(ALTS))
def test_alignment_matches_records(dataset):
    if not records_path(dataset, 7).exists():
        pytest.skip(f"no OM-LEU run for {dataset} (run tag {DEFAULT_RUN_TAGS[dataset]})")
    ds = load_dataset(dataset, 7)
    assert ds.J == len(ALTS[dataset])
    for name, s in ds.splits.items():
        assert s.X.shape == (s.n, ds.J, ds.F)
        assert s.Z.shape == (s.n, ds.P)
        assert np.isfinite(s.X).all() and np.isfinite(s.Z).all()
        assert s.y.min() >= 0 and s.y.max() < ds.J
        # time column is positive for at least one alternative in nearly every event
        # (a handful of Optima loops have missing PT and car times coded as 0)
        assert (s.X[:, :, 0] > 0).any(axis=1).mean() > 0.95
    # standardisation on train
    assert np.allclose(ds.splits["train"].Z.mean(0), 0, atol=1e-4)
    if dataset == "optima":
        assert ds.splits["train"].I is not None and ds.splits["train"].I.shape[1] == 7
