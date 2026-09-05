"""Minimal torch scaffolding shared by the choice models under ``methods/``.

A :class:`ChoiceModel` exposes ``utilities(batch) -> (N, J)`` (or overrides
``log_prob`` / ``predict_proba`` when the likelihood is simulated).
:func:`fit` runs Adam with early stopping on validation NLL and restores
the best parameters.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ChoiceDataset


@dataclass
class Batch:
    name: str
    X: torch.Tensor      # (N, J, F) raw units
    Xs: torch.Tensor     # (N, J, F) standardised
    Z: torch.Tensor      # (N, P)
    y: torch.Tensor      # (N,)
    I: Optional[torch.Tensor]        # (N, Q) or None
    person: torch.Tensor             # (N,) global person index
    idx: torch.Tensor                # (N,) event index within the split

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    def subset(self, sel: torch.Tensor) -> "Batch":
        return Batch(
            name=self.name, X=self.X[sel], Xs=self.Xs[sel], Z=self.Z[sel], y=self.y[sel],
            I=None if self.I is None else self.I[sel], person=self.person[sel], idx=self.idx[sel],
        )


def person_index(ds: ChoiceDataset) -> Dict[str, int]:
    people = sorted({p for s in ds.splits.values() for p in s.person})
    return {p: i for i, p in enumerate(people)}


def to_batch(ds: ChoiceDataset, split: str, pidx: Dict[str, int]) -> Batch:
    s = ds.splits[split]
    return Batch(
        name=split,
        X=torch.as_tensor(s.X, dtype=torch.float32),
        Xs=torch.as_tensor(ds.X_std(split), dtype=torch.float32),
        Z=torch.as_tensor(s.Z, dtype=torch.float32),
        y=torch.as_tensor(s.y, dtype=torch.long),
        I=None if s.I is None else torch.as_tensor(s.I, dtype=torch.float32),
        person=torch.as_tensor([pidx[p] for p in s.person], dtype=torch.long),
        idx=torch.arange(s.n),
    )


class ChoiceModel(nn.Module):
    """Base class: subclasses implement :meth:`utilities`."""

    panel = False  # True → the loss needs whole persons; no minibatching

    def utilities(self, b: Batch) -> torch.Tensor:  # (N, J)
        raise NotImplementedError

    def log_prob(self, b: Batch) -> torch.Tensor:  # (N,)
        return F.log_softmax(self.utilities(b), dim=1).gather(1, b.y[:, None]).squeeze(1)

    def loss(self, b: Batch) -> torch.Tensor:
        return -self.log_prob(b).mean()

    @torch.no_grad()
    def predict_proba(self, b: Batch) -> np.ndarray:
        self.eval()
        return F.softmax(self.utilities(b), dim=1).cpu().numpy()


def mlp(n_in: int, hidden: int, n_out: int, depth: int = 2, dropout: float = 0.1) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = hidden
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@dataclass
class FitInfo:
    best_epoch: int
    best_val_nll: float
    epochs_run: int
    seconds: float


def fit(
    model: ChoiceModel,
    train: Batch,
    val: Batch,
    *,
    lr: float = 1e-2,
    max_epochs: int = 3000,
    patience: int = 200,
    weight_decay: float = 0.0,
    batch_size: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
) -> FitInfo:
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = math.inf
    best_state = copy.deepcopy(model.state_dict())
    best_epoch, bad = 0, 0
    t0 = time.time()
    gen = torch.Generator().manual_seed(seed)
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        if batch_size is None or model.panel:
            opt.zero_grad()
            loss = model.loss(train)
            loss.backward()
            opt.step()
        else:
            perm = torch.randperm(train.n, generator=gen)
            for start in range(0, train.n, batch_size):
                sel = perm[start:start + batch_size]
                opt.zero_grad()
                loss = model.loss(train.subset(sel))
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            v = float(model.loss(val))
        if v < best - 1e-6:
            best, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if verbose and epoch % 50 == 0:
            print(f"  epoch {epoch} train {float(loss):.4f} val {v:.4f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return FitInfo(best_epoch=best_epoch, best_val_nll=best, epochs_run=epoch, seconds=time.time() - t0)
