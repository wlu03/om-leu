"""Numeric channel (existing Stage 1 + Stage 2), fitted inside one partition.

Returns raw utilities; the temperature is applied later, exactly once, by
``omleu_experiments.calibrate``.  LightGBM runs in its own process (the existing
``experiments.models.boost`` child), which is required on this machine because
LightGBM and PyTorch cannot share a process here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.models.omleu2 import boost_stage, structural_stage

from .views import fold_view, restandardise_covariates

DEFAULT_TAU_GRID = (0.6, 0.8, 1.0)
LOGIT_FLOOR = 40.0     # log-probabilities below e^-40 are not distinguishable here


@dataclass
class NumericFit:
    logits: np.ndarray            # (n_predict, J) raw utilities, no temperature
    info: Dict


def fit_numeric(b: Bundle, *, fit_rows, val_rows, predict_rows, seed: int, view_tag: str,
                person_effects: bool, oof_folds: int = 5, tau_grid=DEFAULT_TAU_GRID,
                use_boost: bool = True) -> NumericFit:
    """Fit Stage 1 (+ Stage 2) on ``fit_rows`` with early stopping on ``val_rows``.

    ``person_effects`` must be False under a person-disjoint protocol: a held-out
    respondent has no fitted effect, and the fallback is the zero effect.
    """
    bv = restandardise_covariates(fold_view(b, fit_rows, val_rows, predict_rows))
    struct = () if person_effects else (("person", False),)
    s1 = structural_stage(bv, seed, oof_folds, struct, view_tag)
    va = bv.idx("val")
    info = {"stage1": s1.info, "person_effects": person_effects}
    if not use_boost:
        U = s1.logits
        info["stage2"] = {"disabled": True}
    else:
        best = (float("inf"), 1.0, torch.zeros(bv.N, bv.J), {})
        trials = {}
        for tau in tau_grid:
            f, binfo = boost_stage(bv, seed, tau * s1.offset, ())
            trials[str(tau)] = {"val_nll": binfo["best_val_nll"], "best_round": binfo["best_round"]}
            if binfo["best_val_nll"] < best[0]:
                best = (binfo["best_val_nll"], tau, f, binfo)
        U = best[1] * s1.logits + best[2]
        info["stage2"] = {"tau": best[1], "trials": trials, "best_round": best[3].get("best_round")}
    rows = torch.as_tensor(np.asarray(predict_rows), dtype=torch.long)
    out = U[rows].detach().double()
    # Numerical guard: centre each event and bound the spread so that no supported input can
    # produce an infinite loss.  The ranking is unchanged; the number of affected events is
    # recorded so that a clamp can never pass silently.
    out = out - out.max(dim=-1, keepdim=True).values
    n_clamped = int((out < -LOGIT_FLOOR).any(-1).sum())
    out = out.clamp_min(-LOGIT_FLOOR)
    info["numerical"] = {"logit_floor": LOGIT_FLOOR, "events_clamped": n_clamped,
                         "events": int(len(rows))}
    arr = out.numpy()
    lp = arr - np.log(np.exp(arr - arr.max(1, keepdims=True)).sum(1, keepdims=True)) - arr.max(1, keepdims=True)
    assert np.isfinite(lp).all(), "numeric channel produced a non-finite log-probability"
    return NumericFit(logits=arr, info=info)
