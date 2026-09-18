"""Anchor pairs for the outcome heads.

A pair is two rendered reference sentences for the same axis that differ on one recorded
quantity whose favourable direction is agreed in advance (less cost, less time, fewer
transfers, less waiting).  The sentences are built from the training partition's own
attribute range, so no evaluation event is used, and they are independent of any choice
label.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from experiments.harness.data import Bundle
from omleu_experiments.contracts import AXES
from omleu_experiments.sentences import _encode

# axis -> (template, low_is_better)
ANCHOR_TEMPLATES = {
    "financial": ("Recorded out-of-pocket cost for this option: {v:.2f}.", True),
    "time": ("Recorded in-vehicle or travel time for this option: {v:.0f} minutes.", True),
    "convenience": ("Recorded access or walking time for this option: {v:.0f} minutes; transfers: 0.", True),
    "reliability": ("Recorded waiting or headway time for this option: {v:.0f} minutes.", True),
}


def build_anchors(b: Bundle, seed: int, n_per_axis: int = 24) -> Dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    tr = b.idx("train").numpy()
    better_txt, worse_txt, axis_idx = [], [], []
    for k, ax in enumerate(AXES):
        if ax not in ANCHOR_TEMPLATES:
            continue
        tpl, low_better = ANCHOR_TEMPLATES[ax]
        col = {"financial": ("cost_chf", "cost_gbp", "cost_100chf"), "time": ("time_h",),
               "convenience": ("walking_h", "access_h"), "reliability": ("waiting_h", "headway_h")}[ax]
        names = b.meta["alt_feature_names"]
        f = next((names.index(c) for c in col if c in names), None)
        if f is None:
            continue
        vals = b.Xnum[tr][:, :, f].reshape(-1).numpy()
        vals = vals[np.isfinite(vals)]
        if vals.max() <= vals.min():
            continue
        lo, hi = np.quantile(vals, 0.1), np.quantile(vals, 0.9)
        for _ in range(n_per_axis):
            a, c = sorted(rng.uniform(lo, hi, 2))
            scale = 60.0 if ax != "financial" else 1.0
            s_lo, s_hi = tpl.format(v=a * scale), tpl.format(v=c * scale)
            better_txt.append(s_lo if low_better else s_hi)
            worse_txt.append(s_hi if low_better else s_lo)
            axis_idx.append(k)
    if not axis_idx:
        return {"E_better": torch.zeros(0, b.K, b.d), "E_worse": torch.zeros(0, b.K, b.d),
                "axis": torch.zeros(0, dtype=torch.long)}
    uniq = sorted(set(better_txt) | set(worse_txt))
    V = _encode(uniq); pos = {t: i for i, t in enumerate(uniq)}
    Eb = torch.zeros(len(axis_idx), b.K, b.d); Ew = torch.zeros(len(axis_idx), b.K, b.d)
    for n, (bt, wt, k) in enumerate(zip(better_txt, worse_txt, axis_idx)):
        Eb[n, k] = torch.from_numpy(V[pos[bt]]); Ew[n, k] = torch.from_numpy(V[pos[wt]])
    return {"E_better": Eb, "E_worse": Ew, "axis": torch.tensor(axis_idx, dtype=torch.long)}
