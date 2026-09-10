"""Bundle views: the existing training code reads partitions from ``Bundle.split``, so a
fold is expressed by rewriting that vector rather than by copying tensors."""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np
import torch

from experiments.harness.data import Bundle


def fold_view(b: Bundle, fit_rows: Sequence[int], val_rows: Sequence[int], predict_rows: Sequence[int]) -> Bundle:
    """A view whose ``train`` is ``fit_rows``, ``val`` is ``val_rows`` and ``test`` is
    ``predict_rows``.  Rows in no group are excluded from every partition."""
    split = torch.full((b.N,), -1, dtype=torch.int8)
    split[torch.as_tensor(np.asarray(fit_rows), dtype=torch.long)] = 0
    split[torch.as_tensor(np.asarray(val_rows), dtype=torch.long)] = 1
    split[torch.as_tensor(np.asarray(predict_rows), dtype=torch.long)] = 2
    ov = set(map(int, fit_rows)) & set(map(int, predict_rows))
    assert not ov, f"{len(ov)} rows are in both the fitting and the predicted partition"
    return dataclasses.replace(b, split=split)


def restandardise_covariates(b: Bundle) -> Bundle:
    """Re-standardise person covariates inside this view's training partition.

    The exported ``Z`` was standardised on the historical chronological training split,
    whose rows are not the training partition of a person-disjoint fold.  Standardisation
    is affine, so re-standardising the exported values with partition statistics equals
    standardising the raw covariates with those statistics.  Attribute standardisation
    (``Bundle.xnum_std``) already reads the view's own training mask.
    """
    tr = b.idx("train")
    mu = b.Z[tr].mean(0, keepdim=True)
    sd = b.Z[tr].std(0, keepdim=True)
    keep = (sd > 1e-6).float()
    # A covariate that is constant across this partition's training rows carries no training
    # information; it is set to zero rather than divided by a near-zero deviation, which would
    # turn a rare category present only in a held-out respondent into an enormous value.
    Z = ((b.Z - mu) / torch.where(sd > 1e-6, sd, torch.ones_like(sd))) * keep
    return dataclasses.replace(b, Z=Z)


def replace_embeddings(b: Bundle, E: torch.Tensor) -> Bundle:
    """A view with a different sentence-embedding tensor (templates, controls, edits)."""
    assert E.shape[:3] == b.E.shape[:3], f"embedding shape {tuple(E.shape)} does not match {tuple(b.E.shape)}"
    return dataclasses.replace(b, E=E.float())


def inner_split(rows: np.ndarray, cluster: np.ndarray, seed: int, val_frac: float = 0.2):
    """Split outer-training rows into fitting and early-stopping partitions by cluster,
    so no cluster informs the stopping rule for its own events."""
    cl = np.array(sorted(set(cluster[rows].tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cl))
    n_val = max(1, int(round(val_frac * len(cl))))
    val_cl = set(cl[perm[:n_val]].tolist())
    is_val = np.array([c in val_cl for c in cluster[rows]])
    return rows[~is_val], rows[is_val]
