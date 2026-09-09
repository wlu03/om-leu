"""Plain feed-forward baseline members: the same LLM consequence sentences (or numeric inputs)
fed to an ordinary MLP with no structured utility, no attribute heads, no person weights, no
attention.  One scalar utility per alternative, softmax over the choice set; five seeds are
averaged in probability exactly like the OM-LEU 2 members, so the comparison isolates the
model design, not the ensembling.

Input blocks per (event i, alternative j), chosen by ``inputs``:
    "sent"       mean over the K sentence embeddings           (d = 768)
    "sent_flat"  the K embeddings concatenated, slot order kept (K * d = 3840)
    "z"          standardised person covariates z_i            (P)
    "x"          standardised alternative attributes, alternative one-hot, history h_ij
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from experiments.harness.data import Bundle
from experiments.models.omleu2 import MEMBER_FACTORIES


class PlainNN(nn.Module):
    has_probe = False

    def __init__(self, b: Bundle, *, inputs: Sequence[str] = ("sent",), hidden: Sequence[int] = (128, 64),
                 dropout: float = 0.2):
        super().__init__()
        self.inputs = tuple(inputs)
        blocks = []
        for name in self.inputs:
            if name == "sent":
                blocks.append(b.E.mean(2))
            elif name == "sent_flat":
                blocks.append(b.E.reshape(b.N, b.J, b.K * b.d))
            elif name == "z":
                blocks.append(b.Z[:, None, :].expand(b.N, b.J, b.P))
            elif name == "x":
                blocks.append(torch.cat([b.xnum_std(), b.alt_onehot(), b.Xhist], -1))
            else:
                raise ValueError(name)
        X = torch.cat(blocks, -1).float()
        self.register_buffer("X", X, persistent=False)
        layers, d_in = [], X.shape[-1]
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def probe_logits(self, b: Bundle, idx: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def utilities(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.X[idx]).squeeze(-1)[:, None, :]           # (n, 1, J) for the unc_features interface

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.mlp(self.X[idx]).squeeze(-1), -1)  # (n, J) log-probabilities


MEMBER_FACTORIES["plain_nn"] = lambda b, **kw: PlainNN(b, **kw)
