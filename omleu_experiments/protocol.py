"""Evaluation protocols.

Primary: person-disjoint (cold start).  Every event of a respondent, and of a household
where that is the sampling unit, stays in one partition; no respondent-specific effect
may be fitted from a held-out respondent's labels.

Secondary: time-respecting.  Only legitimate calendar ordering qualifies; a
stated-preference task index or a date manufactured by a preparation script does not.
:func:`ordering_support` decides this from the data and returns a reasoned status.

Historical protocol: the chronological within-person split used by the existing
results is kept under an explicitly named reproduction configuration, because its test
respondents also appear in training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import stable_hash


@dataclass(frozen=True)
class Partition:
    """Row indices of one protocol instance.  ``dev_folds`` are the outer folds used for
    held-out development predictions; ``test`` is opened only after configurations freeze."""

    protocol: str
    train_dev: np.ndarray
    test: np.ndarray
    dev_folds: List[np.ndarray]
    cluster_of_row: np.ndarray
    notes: Dict[str, str]

    @property
    def manifest_hash(self) -> str:
        return stable_hash({"p": self.protocol, "test": sorted(map(int, self.test)),
                            "folds": [sorted(map(int, f)) for f in self.dev_folds]})


def ordering_support(order_values: Sequence[float], order_kind: str, person: Sequence[str]) -> Dict[str, object]:
    """Decide whether a time-respecting protocol is supported by this ordering field."""
    ov = np.asarray(order_values, dtype=float)
    n_distinct = int(len(np.unique(ov)))
    per_person = {}
    for p, v in zip(person, ov):
        per_person.setdefault(p, set()).add(float(v))
    max_tie_block = int(np.max(np.unique(ov, return_counts=True)[1])) if len(ov) else 0
    supported = order_kind == "calendar" and n_distinct > 10 * len(set(person)) ** 0.0 and n_distinct > 50
    reason = {
        "calendar": "real timestamps" if supported else "too few distinct timestamps to define blocks",
        "task_index": "stated-preference task order is presentation order, not calendar time",
        "synthetic": "dates were manufactured by a preparation script",
        "none": "no ordering field",
    }[order_kind]
    return {"order_kind": order_kind, "n_distinct": n_distinct, "max_tie_block": max_tie_block,
            "supported": bool(supported), "reason": reason}


def person_disjoint(person: Sequence[str], cluster: Sequence[str], master_seed: int,
                    *, test_frac: float = 0.2, n_folds: int = 5) -> Partition:
    """Split clusters into a locked test set and ``n_folds`` development folds."""
    cl = np.asarray(cluster)
    uniq = np.array(sorted(set(cl.tolist())))
    rng = np.random.default_rng(master_seed)
    perm = rng.permutation(len(uniq))
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_cl = set(uniq[perm[:n_test]].tolist())
    dev_cl = uniq[perm[n_test:]]
    fold_of_cluster = {c: i % n_folds for i, c in enumerate(dev_cl[rng.permutation(len(dev_cl))])}
    rows = np.arange(len(cl))
    test_rows = rows[np.array([c in test_cl for c in cl])]
    dev_rows = rows[np.array([c not in test_cl for c in cl])]
    folds = [dev_rows[np.array([fold_of_cluster[c] == k for c in cl[dev_rows]])] for k in range(n_folds)]
    assert sum(len(f) for f in folds) == len(dev_rows)
    assert not (set(cl[test_rows]) & set(cl[dev_rows])), "cluster appears in both test and development"
    return Partition("person_disjoint", dev_rows, test_rows, folds, cl,
                     {"unit": "household_or_person", "test_clusters": str(len(test_cl)),
                      "note": "no respondent-specific effect may be fitted for a held-out respondent"})


def rolling_origin(order_values: Sequence[float], cluster: Sequence[str], *, n_folds: int = 5,
                   test_frac: float = 0.2, warmup_frac: float = 0.2) -> Partition:
    """Forward-only blocks on a legitimate calendar ordering.

    Tied events stay together.  The earliest ``warmup_frac`` of development events is
    excluded from out-of-fold calibration because no earlier block can predict it.
    """
    ov = np.asarray(order_values, dtype=float)
    order = np.argsort(ov, kind="stable")
    n = len(ov)
    n_test = int(round(test_frac * n))
    cut = ov[order[n - n_test]]
    test_rows = np.where(ov >= cut)[0]
    dev_rows = np.where(ov < cut)[0]
    dev_sorted = dev_rows[np.argsort(ov[dev_rows], kind="stable")]
    warm = int(round(warmup_frac * len(dev_sorted)))
    blocks = np.array_split(dev_sorted[warm:], n_folds)
    return Partition("temporal_rolling_origin", dev_rows, test_rows, [np.sort(b) for b in blocks],
                     np.asarray(cluster),
                     {"cutoff": str(cut), "warmup_events_excluded_from_calibration": str(warm),
                      "cutoff_kind": "global", "task": "predict later events from earlier ones",
                      "note": "fold k is predicted by a model fitted only on events before its first timestamp"})


def historical_chronological(split_codes: Sequence[int], cluster: Sequence[str], n_folds: int = 5) -> Partition:
    """Reproduction of the existing within-person chronological split.

    Test respondents also appear in training, so this is a reproduction configuration
    only; it is never used for the primary claims.
    """
    sc = np.asarray(split_codes)
    rows = np.arange(len(sc))
    dev = rows[sc != 2]
    test = rows[sc == 2]
    rng = np.random.default_rng(0)
    folds = np.array_split(dev[rng.permutation(len(dev))], n_folds)
    return Partition("historical_chronological", dev, test, [np.sort(f) for f in folds], np.asarray(cluster),
                     {"warning": "test respondents also appear in training; not person-disjoint"})
