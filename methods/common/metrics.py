"""Metrics for ``methods/`` — the same functions the OM-LEU pipeline uses."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from src.eval.metrics import compute_all


def evaluate(probs: np.ndarray, y: np.ndarray, *, n_params: int, n_train: int) -> Dict[str, Any]:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)
    logits = np.log(p)
    raw = asdict(compute_all(logits, y, n_params=n_params, n_train=n_train))
    # normalise names: top1_val -> top1, nll_val -> nll, ...
    m = {(k[:-4] if k.endswith("_val") else k): v for k, v in raw.items()}
    p_chosen = p[np.arange(len(y)), y]
    m["per_event_nll"] = (-np.log(p_chosen)).tolist()
    m["per_event_top1"] = (p.argmax(1) == y).astype(int).tolist()
    return m
