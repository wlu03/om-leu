"""E2: axis-preserving outcome scores with a separate person valuation.

    h[i,j,k] = LayerNorm(W e[i,j,k])          shared projection, per sentence slot
    c[i,j,k] = g_k(h[i,j,k])                  one head per axis, reading only its own axis text
    w[i,:]   = softmax(f(z_i, history_i))     valuation weights, no text
    u[i,j]   = s_i * sum_k w[i,k] c[i,j,k]
    q[i,:]   = masked_softmax(u[i,:])

Axes are never pooled before their scores are computed, and an outcome head never sees a
numeric covariate.  Higher ``c`` means a more favourable outcome *level* on that axis as
the anchors define it; it is not an observed satisfaction and is not identified as a
psychological quantity without the anchoring evidence.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from omleu_experiments.readers import SemanticReader

SENSITIVITY = ("fixed", "global", "person")


class AxisNet(nn.Module):
    def __init__(self, b: Bundle, *, r: int = 32, sensitivity: str = "global", head_hidden: int = 0,
                 w_hidden: int = 32, slot_drop: float = 0.0, use_history: bool = True,
                 score_clip: float = 6.0, s_max: float = 4.0, share_head: bool = False):
        super().__init__()
        assert sensitivity in SENSITIVITY, sensitivity
        self.K, self.r, self.sensitivity = b.K, r, sensitivity
        self.slot_drop, self.score_clip, self.s_max = slot_drop, score_clip, s_max
        self.use_history = use_history
        self.proj = nn.Linear(b.d, r, bias=False)
        self.norm = nn.LayerNorm(r)
        def mk() -> nn.Module:
            if head_hidden > 0:
                return nn.Sequential(nn.Linear(r, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
            return nn.Linear(r, 1)
        self.share_head = share_head
        self.heads = nn.ModuleList([mk() for _ in range(1 if share_head else b.K)])
        pz = b.P + (2 * b.J if use_history else 0)
        self.val = nn.Sequential(nn.Linear(pz, w_hidden), nn.ReLU(), nn.Linear(w_hidden, b.K))
        if sensitivity == "global":
            self.log_s = nn.Parameter(torch.zeros(()))
        elif sensitivity == "person":
            self.s_net = nn.Sequential(nn.Linear(pz, w_hidden), nn.ReLU(), nn.Linear(w_hidden, 1))

    # ---- pieces -------------------------------------------------------------------
    def person_input(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        z = b.Z[idx]
        if not self.use_history:
            return z
        return torch.cat([z, b.Xhist[idx].reshape(len(idx), -1)], -1)

    def scores(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        """(n, J, K) per-axis outcome scores; head k sees only slot k."""
        H = self.norm(self.proj(b.E[idx]))                       # (n,J,K,r)
        cols = [self.heads[0 if self.share_head else k](H[:, :, k, :]) for k in range(self.K)]
        c = torch.cat(cols, -1)                                  # (n,J,K)
        return torch.clamp(c, -self.score_clip, self.score_clip)

    def weights(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.val(self.person_input(b, idx)), -1)      # (n,K), simplex

    def sensitivity_of(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        if self.sensitivity == "fixed":
            return torch.ones(len(idx), device=b.Z.device)
        if self.sensitivity == "global":
            return torch.exp(self.log_s).clamp(max=self.s_max).expand(len(idx))
        return self.s_max * torch.sigmoid(self.s_net(self.person_input(b, idx)).squeeze(-1))

    def utilities(self, b: Bundle, idx: torch.Tensor, *, scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        c = self.scores(b, idx) if scores is None else scores
        w = self.weights(b, idx)
        if self.training and self.slot_drop > 0:
            m = (torch.rand_like(w) > self.slot_drop).float()
            m = torch.where(m.sum(-1, keepdim=True) > 0, m, torch.ones_like(m))
            w = (w * m) / (w * m).sum(-1, keepdim=True).clamp_min(1e-9)
        s = self.sensitivity_of(b, idx)
        return s[:, None] * (c * w[:, None, :]).sum(-1)                    # (n,J)

    def forward(self, b: Bundle, idx: torch.Tensor, avail: Optional[torch.Tensor] = None) -> torch.Tensor:
        u = self.utilities(b, idx)
        if avail is not None:
            u = u.masked_fill(~avail, -1e30)
        return torch.log_softmax(u, -1)

    def probe_logits(self, b, idx):
        return None

    # ---- diagnostics ---------------------------------------------------------------
    def factors(self, b: Bundle, idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            c, w, s = self.scores(b, idx), self.weights(b, idx), self.sensitivity_of(b, idx)
            return {"scores": c, "weights": w, "sensitivity": s, "utilities": self.utilities(b, idx, scores=c)}


def anchor_loss(model: AxisNet, b: Bundle, anchors: Dict[str, torch.Tensor], margin: float = 0.5) -> torch.Tensor:
    """Ordering loss on independent reference examples.

    ``anchors`` holds ``E_better`` and ``E_worse`` of shape (n, K, d) plus ``axis`` indices:
    each pair differs on one recorded quantity whose direction is known, so the better
    example must score at least ``margin`` above the worse one on that axis.  Satisfying
    sampled pairs does not establish a global ordering guarantee.
    """
    Hb = model.norm(model.proj(anchors["E_better"]))
    Hw = model.norm(model.proj(anchors["E_worse"]))
    k = anchors["axis"]
    cb = torch.stack([model.heads[0 if model.share_head else int(kk)](Hb[n, kk]).squeeze(-1) for n, kk in enumerate(k)])
    cw = torch.stack([model.heads[0 if model.share_head else int(kk)](Hw[n, kk]).squeeze(-1) for n, kk in enumerate(k)])
    return F.relu(margin - (cb - cw)).mean()


class AxisReader(SemanticReader):
    """Five-member axis-preserving reader, ensembled in probability like the baseline."""

    name = "axis"

    def __init__(self, *, anchors: bool = False, anchor_weight: float = 0.3, **kw):
        super().__init__(**kw)
        self.use_anchors, self.anchor_weight = anchors, anchor_weight
        self._anchor_batch = None

    def _member(self, b):
        return AxisNet(b, **self.kw)

    def _aux_loss(self, m):
        if not self.use_anchors or self._anchor_batch is None:
            return None
        aw, batch = self.anchor_weight, self._anchor_batch
        return lambda _m, _b, _sel: aw * anchor_loss(m, _b, batch)

    def fit(self, b: Bundle, seed: int) -> None:
        if self.use_anchors:
            from .anchors import build_anchors
            self._anchor_batch = build_anchors(b, seed)
        super().fit(b, seed)
        if self.use_anchors:
            self.info["anchor_pairs"] = int(len(self._anchor_batch["axis"]))
