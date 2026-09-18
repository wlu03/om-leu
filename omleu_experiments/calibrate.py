"""Mixture calibration: p = (1-pi) * softmax(a * u_numeric) + pi * q_semantic.

``a`` is the inverse temperature, so the numeric temperature is ``T = 1/a``.  The
temperature is applied exactly once, inside :func:`mixture_logprob`; callers must pass
raw numeric logits, never pre-scaled ones.  Everything is computed in log space.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


def mixture_logprob(numeric_logits: torch.Tensor, semantic_logprobs: Optional[torch.Tensor],
                    gamma: torch.Tensor, log_a: torch.Tensor,
                    avail: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Log-probabilities of the calibrated mixture.  ``numeric_logits`` are raw."""
    u = numeric_logits * torch.exp(log_a)
    if avail is not None:
        u = u.masked_fill(~avail, -1e30)
    lp_num = torch.log_softmax(u, -1)
    if semantic_logprobs is None:
        return lp_num
    pi = torch.sigmoid(gamma)
    return torch.logaddexp(torch.log1p(-pi + EPS) + lp_num, torch.log(pi + EPS) + semantic_logprobs)


def _nll(numeric_logits, semantic_logprobs, y, gamma, log_a, avail=None) -> torch.Tensor:
    lp = mixture_logprob(numeric_logits, semantic_logprobs, gamma, log_a, avail)
    return -lp[torch.arange(len(y)), y].mean()


def fit_mixture(numeric_logits: np.ndarray, semantic_logprobs: Optional[np.ndarray], y: np.ndarray,
                *, avail: Optional[np.ndarray] = None, fit_pi: bool = True, fit_temp: bool = True,
                grid_check: bool = True) -> Dict[str, float]:
    """Fit ``(pi, a)`` by L-BFGS on held-out development predictions.

    Returns the fitted scalars, the development NLL, the pi=0 and pi=1 boundary NLLs and
    a grid diagnostic (best NLL on a coarse two-dimensional grid).  These development
    predictions must not afterwards be reported as an evaluation of the fitted mixture.
    """
    U = torch.as_tensor(numeric_logits, dtype=torch.float64)
    Q = None if semantic_logprobs is None else torch.as_tensor(semantic_logprobs, dtype=torch.float64)
    Y = torch.as_tensor(y, dtype=torch.long)
    A = None if avail is None else torch.as_tensor(avail, dtype=torch.bool)
    gamma = torch.tensor(-6.0, dtype=torch.float64, requires_grad=fit_pi and Q is not None)
    log_a = torch.tensor(0.0, dtype=torch.float64, requires_grad=fit_temp)
    params = [p for p in (gamma, log_a) if p.requires_grad]
    if params:
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = _nll(U, Q, Y, gamma, log_a, A)
            loss.backward()
            return loss
        opt.step(closure)
    with torch.no_grad():
        dev_nll = float(_nll(U, Q, Y, gamma, log_a, A))
        out = {"pi": float(torch.sigmoid(gamma)), "a": float(torch.exp(log_a)),
               "temperature": float(1.0 / torch.exp(log_a)), "dev_nll": dev_nll,
               "n_dev_events": int(len(y))}
        if Q is not None:
            out["dev_nll_pi0"] = float(_nll(U, Q, Y, torch.tensor(-40.0, dtype=torch.float64), log_a, A))
            out["dev_nll_pi1"] = float(_nll(U, Q, Y, torch.tensor(40.0, dtype=torch.float64), log_a, A))
            assert np.isfinite(out["dev_nll_pi0"]) and np.isfinite(out["dev_nll_pi1"]), "boundary NLL not finite"
        if grid_check and Q is not None:
            best = np.inf; arg = (np.nan, np.nan)
            for pi_g in np.linspace(0.0, 1.0, 21):
                g = torch.tensor(float(np.log(max(pi_g, 1e-9) / max(1 - pi_g, 1e-9))), dtype=torch.float64)
                for la in np.linspace(-1.0, 1.0, 21):
                    v = float(_nll(U, Q, Y, g, torch.tensor(float(la), dtype=torch.float64), A))
                    if v < best:
                        best, arg = v, (float(pi_g), float(np.exp(la)))
            out["grid_best_nll"] = best; out["grid_best_pi"] = arg[0]; out["grid_best_a"] = arg[1]
            out["grid_minus_lbfgs"] = best - dev_nll
    return out


def apply_calibration(numeric_logits: np.ndarray, semantic_logprobs: Optional[np.ndarray],
                      cal: Dict[str, float], avail: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply fitted ``(pi, a)`` once to produce final log-probabilities."""
    U = torch.as_tensor(numeric_logits, dtype=torch.float64)
    Q = None if semantic_logprobs is None else torch.as_tensor(semantic_logprobs, dtype=torch.float64)
    A = None if avail is None else torch.as_tensor(avail, dtype=torch.bool)
    gamma = torch.tensor(float(np.log(max(cal["pi"], 1e-12) / max(1 - cal["pi"], 1e-12))), dtype=torch.float64)
    log_a = torch.tensor(float(np.log(cal["a"])), dtype=torch.float64)
    with torch.no_grad():
        return mixture_logprob(U, Q, gamma, log_a, A).numpy()


# --------------------------------------------------------------------------------------
# Conditional mixture gate (used by E7).  The gate is a regularised logistic function of a
# small, predeclared context vector, shrunk towards the global mixture weight.  It never
# sees a label, the correctness of a channel, or any test-set quantity.
# --------------------------------------------------------------------------------------

def fit_conditional_gate(numeric_logits: np.ndarray, semantic_logprobs: np.ndarray, y: np.ndarray,
                         context: np.ndarray, *, l2: float = 1.0, avail: Optional[np.ndarray] = None,
                         fit_temp: bool = True) -> Dict[str, float]:
    """pi_i = sigmoid(gamma + beta^T context_i), with ``l2`` shrinking beta towards zero,
    i.e. towards the global gate.  Fitted on held-out development predictions only."""
    U = torch.as_tensor(numeric_logits, dtype=torch.float64)
    Q = torch.as_tensor(semantic_logprobs, dtype=torch.float64)
    C = torch.as_tensor(context, dtype=torch.float64)
    Y = torch.as_tensor(y, dtype=torch.long)
    A = None if avail is None else torch.as_tensor(avail, dtype=torch.bool)
    gamma = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
    beta = torch.zeros(C.shape[1], dtype=torch.float64, requires_grad=True)
    log_a = torch.tensor(0.0, dtype=torch.float64, requires_grad=fit_temp)
    params = [gamma, beta] + ([log_a] if fit_temp else [])
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def loss_fn():
        pi = torch.sigmoid(gamma + C @ beta)[:, None]
        u = U * torch.exp(log_a)
        if A is not None:
            u = u.masked_fill(~A, -1e30)
        lp = torch.logaddexp(torch.log1p(-pi + EPS) + torch.log_softmax(u, -1), torch.log(pi + EPS) + Q)
        return -lp[torch.arange(len(Y)), Y].mean() + l2 * (beta ** 2).sum() / len(Y)

    def closure():
        opt.zero_grad(); l = loss_fn(); l.backward(); return l
    opt.step(closure)
    with torch.no_grad():
        pi = torch.sigmoid(gamma + C @ beta)
        return {"gamma": float(gamma), "beta": [float(x) for x in beta], "a": float(torch.exp(log_a)),
                "temperature": float(1.0 / torch.exp(log_a)), "l2": l2, "dev_nll": float(loss_fn()),
                "pi_mean": float(pi.mean()), "pi_p10": float(pi.quantile(0.1)), "pi_p90": float(pi.quantile(0.9)),
                "n_dev_events": int(len(y))}


def apply_conditional_gate(numeric_logits: np.ndarray, semantic_logprobs: np.ndarray, context: np.ndarray,
                           cal: Dict[str, float], avail: Optional[np.ndarray] = None) -> np.ndarray:
    U = torch.as_tensor(numeric_logits, dtype=torch.float64)
    Q = torch.as_tensor(semantic_logprobs, dtype=torch.float64)
    C = torch.as_tensor(context, dtype=torch.float64)
    A = None if avail is None else torch.as_tensor(avail, dtype=torch.bool)
    with torch.no_grad():
        pi = torch.sigmoid(torch.tensor(cal["gamma"], dtype=torch.float64) +
                           C @ torch.as_tensor(cal["beta"], dtype=torch.float64))[:, None]
        u = U * cal["a"]
        if A is not None:
            u = u.masked_fill(~A, -1e30)
        return torch.logaddexp(torch.log1p(-pi + EPS) + torch.log_softmax(u, -1),
                               torch.log(pi + EPS) + Q).numpy()
