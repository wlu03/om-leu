"""Consistency losses on the common outcome-score interface.

A model participates by exposing ``scores(bundle, rows) -> (n, J, K)``.  The losses are
defined against that interface only, so they can be tested with a stub before the axis
model is merged.  Margins are expressed in score units, which are bounded by the model's
own clipping, so a margin has a stated scale.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F


def paraphrase_loss(scores_a: torch.Tensor, scores_b: torch.Tensor) -> torch.Tensor:
    """Squared drift between the scores of a claim-preserving paraphrase pair."""
    return ((scores_a - scores_b) ** 2).mean()


def order_loss(scores_original: torch.Tensor, scores_edited: torch.Tensor, axes: Sequence[int],
               slot: int, margin: float = 0.0) -> torch.Tensor:
    """The edited alternative must be no better on the affected axes.

    With ``margin = 0`` this only forbids an improvement; a positive margin demands a
    strict worsening and is used only when the edit's size justifies one.
    """
    d = scores_edited[:, slot, axes] - scores_original[:, slot, axes]
    return F.relu(d + margin).mean()


def local_loss(scores_original: torch.Tensor, scores_edited: torch.Tensor, axes: Sequence[int],
               slot: int) -> torch.Tensor:
    """Axes the edit cannot touch must not move."""
    if len(axes) == 0:
        return scores_original.new_zeros(())
    d = scores_edited[:, slot, axes] - scores_original[:, slot, axes]
    return (d ** 2).mean()


def consistency_losses(model, b_orig, b_edit, b_para, rows, *, slot: int, affected, unaffected,
                       lam_para: float = 0.0, lam_order: float = 0.0, lam_local: float = 0.0) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if lam_para > 0 and b_para is not None:
        out["para"] = lam_para * paraphrase_loss(model.scores(b_orig, rows), model.scores(b_para, rows))
    if (lam_order > 0 or lam_local > 0) and b_edit is not None:
        s0, s1 = model.scores(b_orig, rows), model.scores(b_edit, rows)
        if lam_order > 0:
            out["order"] = lam_order * order_loss(s0, s1, affected, slot)
        if lam_local > 0:
            out["local"] = lam_local * local_loss(s0, s1, unaffected, slot)
    return out


def collapse_diagnostics(scores: torch.Tensor) -> Dict[str, float]:
    """Representation collapse check: a constant score satisfies every consistency loss."""
    return {"score_sd_over_events": float(scores.std(0).mean()),
            "score_sd_over_alternatives": float(scores.std(1).mean()),
            "score_sd_over_axes": float(scores.std(2).mean())}
