"""Training / evaluation utilities shared by all variants.

A variant is an ``nn.Module`` with ``forward(bundle, idx) -> logits (n, J)``
(it reads whatever tensors it needs from the bundle at the given row indices)
and optionally ``prefit(bundle)`` for a stage-1 fit of its structural part.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Bundle


@dataclass
class FitResult:
    best_epoch: int
    best_val_nll: float
    epochs: int
    seconds: float
    val_curve: list


def nll_on(model: nn.Module, b: Bundle, idx: torch.Tensor, batch: int = 2048) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            sel = idx[s:s + batch]
            lg = model(b, sel)
            tot += float(F.cross_entropy(lg, b.y[sel], reduction="sum")); n += len(sel)
    return tot / n


def fit(model: nn.Module, b: Bundle, *, lr: float = 1e-3, max_epochs: int = 100, patience: int = 10,
        batch_size: int = 256, weight_decay: float = 1e-4, seed: int = 0, extra_loss: Optional[Callable] = None,
        params: Optional[list] = None, verbose: bool = False) -> FitResult:
    """Adam with early stopping on validation NLL; restores the best state."""
    torch.manual_seed(seed)
    tr, va = b.idx("train"), b.idx("val")
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, best_ep, bad, curve = math.inf, copy.deepcopy(model.state_dict()), 0, 0, []
    t0 = time.time()
    g = torch.Generator().manual_seed(seed)
    v0 = nll_on(model, b, va); curve.append(v0)
    if v0 < best:
        best, best_ep = v0, 0
    ep = 0
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = tr[torch.randperm(len(tr), generator=g)]
        for s in range(0, len(perm), batch_size):
            sel = perm[s:s + batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(b, sel), b.y[sel])
            if extra_loss is not None:
                loss = loss + extra_loss(model, b, sel)
            loss.backward()
            opt.step()
        v = nll_on(model, b, va); curve.append(v)
        if verbose:
            print(f"  epoch {ep} val {v:.4f}")
        if v < best - 1e-5:
            best, best_ep, bad = v, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(best_epoch=best_ep, best_val_nll=best, epochs=ep, seconds=time.time() - t0, val_curve=curve)


def lbfgs_prefit(loss_fn: Callable[[], torch.Tensor], params: list, max_iter: int = 500) -> float:
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-9,
                            history_size=50, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = loss_fn()
        l.backward()
        return l
    opt.step(closure)
    return float(loss_fn())


def evaluate(model: nn.Module, b: Bundle, split: str = "test") -> Dict:
    from src.eval.metrics import compute_all
    idx = b.idx(split)
    model.eval()
    with torch.no_grad():
        lg = torch.cat([model(b, idx[s:s + 2048]) for s in range(0, len(idx), 2048)], 0)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    m = asdict(compute_all(lg, b.y[idx], n_params=n_params, n_train=len(b.idx("train"))))
    m = {(k[:-4] if k.endswith("_val") else k): v for k, v in m.items()}
    p = torch.softmax(lg, 1)
    pc = p[torch.arange(len(idx)), b.y[idx]].clamp_min(1e-12)
    m["per_event_nll"] = (-pc.log()).tolist()
    m["per_event_top1"] = (lg.argmax(1) == b.y[idx]).int().tolist()
    return m
