"""Compositional holdouts built from pre-choice fields only.

A cell is a combination of two pre-choice features, for example trip purpose and whether the
choice set contains a transfer burden.  Definitions are frozen from development features
before any test metric is read, person disjointness is enforced when it is claimed, and a
cell with too little support is refused rather than reported.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from experiments.harness.data import Bundle


def compositional_holdout(b: Bundle, dev_rows: np.ndarray, clusters: np.ndarray, *, feature_a: str,
                          feature_b: str, min_support: int = 60) -> Dict[str, object]:
    """Return the row mask of one held-out combination and its support statistics."""
    z = b.meta["z_names"]
    if feature_a not in z:
        return {"status": "blocked_data", "reason": f"{feature_a} is not a covariate of {b.dataset}"}
    A = b.Z[:, z.index(feature_a)].numpy() > 0
    names = b.meta["alt_feature_names"]
    if feature_b in names:
        B = b.Xnum[:, :, names.index(feature_b)].sum(1).numpy() > 0
    elif feature_b in z:
        B = b.Z[:, z.index(feature_b)].numpy() > 0
    else:
        return {"status": "blocked_data", "reason": f"{feature_b} is not available in {b.dataset}"}
    cell = A & B
    held = np.intersect1d(np.where(cell)[0], dev_rows)
    rest = np.setdiff1d(dev_rows, held)
    if len(held) < min_support or len(rest) < min_support:
        return {"status": "blocked_data", "reason": f"support {len(held)} / {len(rest)} below {min_support}"}
    overlap = set(clusters[held].tolist()) & set(clusters[rest].tolist())
    return {"status": "completed", "held_rows": held, "rest_rows": rest,
            "cell": f"{feature_a} x {feature_b}", "n_held": int(len(held)), "n_rest": int(len(rest)),
            "cluster_overlap": int(len(overlap)),
            "person_disjoint": bool(not overlap)}
