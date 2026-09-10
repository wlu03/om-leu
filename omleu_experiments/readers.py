"""Semantic channels.  Every reader maps sentence embeddings (and, where declared,
person covariates and admissible history) to log-probabilities over the choice set.

``PreservedReader`` is the existing Stage 3, unchanged, so that every comparison has the
baseline architecture in it.  ``PlainReader`` is the ordinary-model control: the same
information through a two-layer network.  ``NumericAuxiliaryReader`` is a
parameter-budget-matched channel that sees no text at all, so that "another predictor
helps" can be separated from "consequences help".
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import fit
from experiments.models.omleu2 import PREF_TRAIN, PREF_LAM_PROBE
from experiments.models.pref import PrefBranch, V1 as PREF_V1, _log_mean_exp

PLAIN_TRAIN = dict(PREF_TRAIN)


class SemanticReader:
    """Interface: ``fit`` on a bundle view whose train/val partitions are the fitting
    partitions, then ``logprobs`` for arbitrary rows of the same view."""

    n_members = 5
    name = "reader"

    def __init__(self, **kw):
        self.kw = kw
        self.members: List[nn.Module] = []
        self.info: Dict = {}

    def _member(self, b: Bundle) -> nn.Module:
        raise NotImplementedError

    def _aux_loss(self, m: nn.Module):
        return None

    def fit(self, b: Bundle, seed: int) -> None:
        torch.manual_seed(seed)
        self.members = [self._member(b) for _ in range(self.n_members)]
        fits = [fit(m, b, params=list(m.parameters()), seed=seed * 100 + i,
                    extra_loss=self._aux_loss(m), **PLAIN_TRAIN) for i, m in enumerate(self.members)]
        self.info = {"members": self.n_members, "best_val": [f.best_val_nll for f in fits],
                     "best_epoch": [f.best_epoch for f in fits],
                     "n_params": int(sum(p.numel() for m in self.members for p in m.parameters())),
                     "seconds": sum(f.seconds for f in fits)}

    def logprobs(self, b: Bundle, rows: Sequence[int]) -> np.ndarray:
        idx = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        outs = []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                sel = idx[s:s + 2048]
                outs.append(_log_mean_exp(torch.stack([m(b, sel) for m in self.members], 1), 1))
        return torch.cat(outs, 0).double().numpy()


class PreservedReader(SemanticReader):
    """The existing Stage 3: 768->32 projection, salience attention, five heads,
    person-conditioned head weights, slot dropout, auxiliary probe."""

    name = "preserved"

    def _member(self, b):
        return PrefBranch(b, **{**PREF_V1, **self.kw})

    def _aux_loss(self, m):
        return lambda _m, _b, sel: PREF_LAM_PROBE * F.cross_entropy(m.probe_logits(_b, sel), _b.y[sel])


class _PlainNet(nn.Module):
    """Two-layer network on a per-alternative feature block; one utility per alternative."""

    def __init__(self, b: Bundle, inputs=("sent",), hidden=(128, 64), dropout=0.2, slots=False):
        super().__init__()
        blocks = []
        for name in inputs:
            if name == "sent":
                blocks.append(b.E.reshape(b.N, b.J, b.K * b.d) if slots else b.E.mean(2))
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

    def forward(self, b, idx):
        return torch.log_softmax(self.mlp(self.X[idx]).squeeze(-1), -1)


class PlainReader(SemanticReader):
    """Ordinary embedding model.  ``slots=True`` keeps the K axis slots separate, so
    that pooling is not a confound when comparing with an axis-preserving reader."""

    name = "plain"

    def _member(self, b):
        return _PlainNet(b, **self.kw)


class NumericAuxiliaryReader(SemanticReader):
    """Parameter-budget-matched auxiliary channel with no text: mixed through the same
    calibration procedure, it measures how much of a gain is 'a second predictor'."""

    name = "numeric_auxiliary"

    def _member(self, b):
        return _PlainNet(b, inputs=("z", "x"), hidden=self.kw.get("hidden", (128, 64)),
                         dropout=self.kw.get("dropout", 0.2))


READERS = {"preserved": PreservedReader, "plain": PlainReader, "numeric_auxiliary": NumericAuxiliaryReader}
