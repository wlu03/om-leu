"""Integrated choice and latent variable (ICLV / hybrid choice) model —
the Biogeme Optima case study: one latent "car-loving attitude" with a
structural equation on socio-demographics and linear measurement equations
on seven Likert indicators, estimated simultaneously with the choice model
by simulated maximum likelihood.

eta_n = gamma . z_n + eps_n,                 eps ~ N(0, 1)        (structural)
I_qn  = a_q + lambda_q eta_n + sigma_q nu,   nu ~ N(0, 1)         (measurement, q = 1..Q)
V_jn  = ASC_j + beta_j . x_jn + gamma_j . z_n + b_j eta_n         (choice; b_0 = 0)
L_n   = (1/R) sum_r  P(y_n | eta_n^r) prod_q phi((I_qn - a_q - lambda_q eta_n^r) / sigma_q) / sigma_q

Prediction on held-out respondents uses the structural equation only
(indicators are not needed at forecast time). Requires a dataset with
indicators, i.e. Optima; other datasets report "n/a".
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common.data import ChoiceDataset
from methods.common.torch_base import Batch, fit, n_params, person_index, to_batch
from methods.mnl.model import MNL, coefficient_table


class ICLV(MNL):
    panel = True  # fixed per-split draws; full batch

    def __init__(self, J: int, F: int, P: int, Q: int, *, R: int = 64, seed: int = 0):
        super().__init__(J, F, P)
        self.R, self.Q = R, Q
        self.gamma = nn.Parameter(torch.zeros(P))
        self.b_lat = nn.Parameter(torch.zeros(J))
        self.a = nn.Parameter(torch.full((Q,), 3.0))
        self.lam = nn.Parameter(torch.ones(Q))
        self.log_sig = nn.Parameter(torch.zeros(Q))
        self._gen = torch.Generator().manual_seed(seed)
        self._draws: dict = {}

    def draws(self, b: Batch) -> torch.Tensor:
        key = (b.name, b.n)
        if key not in self._draws:
            self._draws[key] = torch.randn(b.n, self.R, generator=self._gen)
        return self._draws[key]

    def eta(self, b: Batch) -> torch.Tensor:  # (N, R)
        return (b.Z @ self.gamma)[:, None] + self.draws(b)

    def utilities_draws(self, b: Batch) -> torch.Tensor:  # (N, R, J)
        base = self.linear_utilities(b)
        eta = self.eta(b)
        return base[:, None, :] + (self.b_lat * self.ref_mask)[None, None, :] * eta[:, :, None]

    def loss(self, b: Batch) -> torch.Tensor:
        V = self.utilities_draws(b)
        eta = self.eta(b)
        lp = F.log_softmax(V, dim=-1).gather(-1, b.y[:, None, None].expand(-1, self.R, 1)).squeeze(-1)  # (N, R)
        if b.I is not None:
            sig = self.log_sig.exp()
            mu = self.a[None, None, :] + self.lam[None, None, :] * eta[:, :, None]          # (N, R, Q)
            obs = ~torch.isnan(b.I)[:, None, :]
            I = torch.nan_to_num(b.I, nan=0.0)[:, None, :]   # masked below; avoids NaN gradients
            z = torch.where(obs, (I - mu) / sig, torch.zeros_like(mu))
            logphi = -0.5 * z ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)
            lp = lp + torch.where(obs, logphi, torch.zeros_like(logphi)).sum(-1)
        ll = torch.logsumexp(lp, dim=1) - math.log(self.R)
        return -ll.mean()

    @torch.no_grad()
    def predict_proba(self, b: Batch):
        self.eval()
        return F.softmax(self.utilities_draws(b), dim=-1).mean(1).cpu().numpy()


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    if not ds.indicator_names:
        return {"skipped": f"{ds.dataset} has no attitudinal indicators; ICLV not applicable"}
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = ICLV(ds.J, ds.F, ds.P, len(ds.indicator_names), R=64, seed=seed)
    info = fit(model, tr, va, lr=0.02, max_epochs=3000, patience=200, weight_decay=1e-6, seed=seed)
    extra = coefficient_table(model, ds)
    extra["latent"] = {
        "b_latent_by_alt": {a: float(v) for a, v in zip(ds.alts, (model.b_lat * model.ref_mask).detach().numpy())},
        "structural_gamma": {z: float(g) for z, g in zip(ds.z_names, model.gamma.detach().numpy())},
        "measurement": {q: {"a": float(a), "lambda": float(l), "sigma": float(s)}
                        for q, a, l, s in zip(ds.indicator_names, model.a.detach().numpy(),
                                              model.lam.detach().numpy(), model.log_sig.exp().detach().numpy())},
    }
    return {"probs_test": model.predict_proba(te), "n_params": n_params(model), "fit": info.__dict__, "extra": extra}
