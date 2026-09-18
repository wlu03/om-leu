"""Metrics and cluster bootstrap.

Primary metric is the mean per-event negative log-likelihood.  Every non-additive
metric is recomputed from saved probabilities, never averaged across saved summaries.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

ECE_BINS = 15


def _probs(logp: np.ndarray) -> np.ndarray:
    p = np.exp(logp - logp.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def event_nll(logp: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -logp[np.arange(len(y)), y]


def brier(logp: np.ndarray, y: np.ndarray, avail: Optional[np.ndarray] = None) -> float:
    """Multiclass Brier score, summed over available alternatives."""
    p = _probs(logp)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(y)), y] = 1.0
    d = (p - onehot) ** 2
    if avail is not None:
        d = d * avail
    return float(d.sum(1).mean())


def top1(logp: np.ndarray, y: np.ndarray) -> float:
    return float((logp.argmax(1) == y).mean())


def ece_top_label(logp: np.ndarray, y: np.ndarray, bins: int = ECE_BINS) -> float:
    """Top-label expected calibration error with fixed equal-width bins on [0, 1]."""
    p = _probs(logp)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def all_metrics(logp: np.ndarray, y: np.ndarray, avail: Optional[np.ndarray] = None) -> Dict[str, float]:
    return {"nll": float(event_nll(logp, y).mean()), "brier": brier(logp, y, avail),
            "top1": top1(logp, y), "ece": ece_top_label(logp, y), "n_events": int(len(y))}


def cluster_draws(clusters: Sequence[str], draws: int = 2000, seed: int = 20260910) -> np.ndarray:
    """Bootstrap draws over independent clusters, identical across variants and seeds.

    Returns an (draws, n_clusters) integer array of cluster positions; a variant maps
    them onto its own events through :func:`paired_diff`.
    """
    uniq = sorted(set(clusters))
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(uniq), (draws, len(uniq)))


def paired_diff(nll_baseline: np.ndarray, nll_variant: np.ndarray, clusters: Sequence[str],
                draws: np.ndarray) -> Dict[str, float]:
    """Paired difference ``NLL_baseline - NLL_variant`` (positive = variant improves),
    with a cluster bootstrap interval.  Events of one cluster always move together."""
    assert len(nll_baseline) == len(nll_variant) == len(clusters)
    uniq = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(uniq)}
    by_cluster = [[] for _ in uniq]
    for i, c in enumerate(clusters):
        by_cluster[pos[c]].append(i)
    d = nll_baseline - nll_variant
    sums = np.array([d[ix].sum() for ix in by_cluster])
    counts = np.array([len(ix) for ix in by_cluster], dtype=float)
    tot = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1e-12)
    return {"mean": float(d.mean()), "lo": float(np.quantile(tot, 0.025)),
            "hi": float(np.quantile(tot, 0.975)), "n_clusters": len(uniq),
            "significant": bool(np.quantile(tot, 0.025) > 0 or np.quantile(tot, 0.975) < 0)}
