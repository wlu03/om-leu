"""Gradient identity and the held-out complementarity diagnostic R'(0)."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch


def gradient_responsibility(p_y: torch.Tensor, q_y: torch.Tensor, pi_train: float) -> torch.Tensor:
    """responsibility = pi q_y / ((1-pi) p_y + pi q_y); the factor by which L_mix scales the
    standalone gradient of -log q_y."""
    return pi_train * q_y / ((1 - pi_train) * p_y + pi_train * q_y)


def r_prime_at_zero(p_y: np.ndarray, q_y: np.ndarray, *, clip: float = None) -> Dict[str, float]:
    """Derivative of the mixture NLL with respect to pi at pi = 0, held out.

    ``R'(0) = 1 - mean(q_y / p_y)``: negative means a small amount of the semantic channel
    lowers the loss.  The ratio is heavy-tailed, so the quantiles and the share of the
    statistic contributed by the largest few events are reported alongside it.  A clipped
    version is reported under its own name and never as the original estimand.
    """
    r = np.asarray(q_y, dtype=float) / np.maximum(np.asarray(p_y, dtype=float), 1e-300)
    out = {"r_prime_at_zero": float(1.0 - r.mean()), "n": int(len(r)),
           "ratio_mean": float(r.mean()), "ratio_median": float(np.median(r)),
           "ratio_q90": float(np.quantile(r, 0.9)), "ratio_q99": float(np.quantile(r, 0.99)),
           "ratio_max": float(r.max())}
    order = np.argsort(-r)
    top = max(1, int(0.01 * len(r)))
    out["share_of_mean_from_top_1pct"] = float(r[order[:top]].sum() / max(r.sum(), 1e-300))
    out["top_events"] = [int(i) for i in order[:10]]
    boot = np.random.default_rng(0).integers(0, len(r), (2000, len(r)))
    means = r[boot].mean(1)
    out["ci_lo"] = float(1.0 - np.quantile(means, 0.975)); out["ci_hi"] = float(1.0 - np.quantile(means, 0.025))
    if clip is not None:
        out[f"r_prime_at_zero_clipped_{clip}"] = float(1.0 - np.minimum(r, clip).mean())
    return out


def finite_difference_r_prime(p: np.ndarray, q: np.ndarray, y: np.ndarray, h: float = 1e-5) -> float:
    """Numerical derivative of the mixture NLL at pi = 0 for the same data."""
    def nll(pi):
        m = (1 - pi) * p + pi * q
        return float(-np.log(m[np.arange(len(y)), y]).mean())
    return (nll(h) - nll(0.0)) / h
