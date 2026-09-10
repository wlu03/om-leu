"""E5: an outcome *distribution* per alternative instead of a point outcome.

    u[i,j] = w[i]^T mu[i,j] - rho[i] * w[i]^T Sigma[i,j] w[i],   rho[i] >= 0

This is a mean-variance approximation of a risk-sensitive valuation, not an identity of
expected utility: it coincides with expected utility only for a quadratic utility or a
Gaussian outcome with an exponential utility.  ``Sigma`` is positive semi-definite by
construction (it is built from a factor ``L`` as ``L L^T``), the overall sensitivity is a
separate parameter from ``rho`` so that scale and risk aversion are not confounded, and
scale normalisation is documented on the class.

Physical variability of an outcome, missing evidence about it, and the model's own
uncertainty are different quantities; only the first belongs in ``Sigma``.  Sampling an
LLM several times measures neither.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def mean_variance_utility(w: torch.Tensor, mu: torch.Tensor, Sigma: torch.Tensor,
                          rho: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
    """(n, J) utilities from (n, K) weights, (n, J, K) means and (n, J, K, K) covariances."""
    lin = (mu * w[:, None, :]).sum(-1)                                  # (n,J)
    quad = torch.einsum("nk,njkl,nl->nj", w, Sigma, w)                  # (n,J) >= 0 for PSD Sigma
    u = lin - rho[:, None] * quad
    return u if s is None else s[:, None] * u


class RiskSensitiveHead(nn.Module):
    """Outcome distribution head: mean vector and a Cholesky factor per alternative.

    Scale convention: ``mu`` is in the same units as the deterministic outcome scores of
    E2 (clipped to +/- ``score_clip``), the diagonal of ``Sigma`` is bounded by
    ``var_max``, and ``rho`` is bounded in ``[0, rho_max]``.  With ``rho = 0`` the head
    reduces exactly to the expected-outcome model.
    """

    def __init__(self, r: int, K: int, *, score_clip: float = 6.0, var_max: float = 4.0,
                 rho_max: float = 2.0, rank: Optional[int] = None):
        super().__init__()
        self.K, self.score_clip, self.var_max, self.rho_max = K, score_clip, var_max, rho_max
        self.rank = rank or K
        self.mu_head = nn.Linear(r, 1)
        self.fac_head = nn.Linear(r, self.rank)
        self.log_diag = nn.Parameter(torch.full((K,), -2.0))

    def mu(self, H: torch.Tensor) -> torch.Tensor:
        """(n, J, K) means from (n, J, K, r) per-axis representations."""
        return torch.clamp(self.mu_head(H).squeeze(-1), -self.score_clip, self.score_clip)

    def sigma(self, H: torch.Tensor) -> torch.Tensor:
        """(n, J, K, K) positive semi-definite covariances, ``L L^T`` plus a bounded diagonal."""
        L = self.fac_head(H)                                            # (n,J,K,rank)
        S = torch.einsum("njkr,njlr->njkl", L, L)
        d = torch.sigmoid(self.log_diag) * self.var_max
        S = S + torch.diag_embed(d.expand(S.shape[:-2] + (self.K,)))
        tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        scale = torch.clamp(self.K * self.var_max / tr.clamp_min(1e-6), max=1.0)
        return S * scale                                                # trace-bounded, still PSD

    def rho(self, n: int, raw: Optional[torch.Tensor] = None) -> torch.Tensor:
        if raw is None:
            return torch.zeros(n)
        return self.rho_max * torch.sigmoid(raw).squeeze(-1)


def data_support_audit(dataset: str) -> Dict[str, object]:
    """Does this dataset support an empirical outcome distribution?

    Requires either repeated observations of the same outcome under the same conditions, or
    independently measured scenario probabilities, or elicited beliefs recorded from the
    respondent.  None of the three mode-choice datasets carries any of them.
    """
    have = {
        "swissmetro": {"repeated_outcomes": False, "scenario_probabilities": False, "elicited_beliefs": False,
                       "note": "stated-preference attributes are fixed scenario values with no distribution"},
        "optima": {"repeated_outcomes": False, "scenario_probabilities": False, "elicited_beliefs": False,
                   "note": "reported trip attributes are single values; the survey has attitudes, not beliefs about delay"},
        "lpmc": {"repeated_outcomes": False, "scenario_probabilities": False, "elicited_beliefs": False,
                 "note": "route-planner attributes are point estimates; no travel-time distribution is distributed with the data"},
    }[dataset]
    have["supported"] = any(v for k, v in have.items() if isinstance(v, bool))
    have["status"] = "completed" if have["supported"] else "blocked_data"
    have["would_enable"] = ("a travel-time or delay distribution per alternative (for example a route "
                            "planner's percentile bands), repeated observations of the same journey, or "
                            "elicited respondent beliefs about delay")
    return have
