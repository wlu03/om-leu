"""Panel mixed logit: the MNL specification of ``methods/mnl`` plus normally
distributed random deviations on the time and cost coefficients, shared by
a person across all of their choice events, estimated by simulated maximum
likelihood with R fixed standard-normal draws per person. Nests the MNL
(sigma = 0), so its fit is never worse in-sample.

beta_{j,time}^r = beta_{j,time} + sigma_t xi_t^r,   beta_{j,cost}^r = beta_{j,cost} + sigma_c xi_c^r
V_j^r = ASC_j + beta_j^r . x_j + gamma_j . z
L_n = (1/R) sum_r prod_{events of person n} P(y | V^r)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common.data import ChoiceDataset, TIME_IDX, COST_IDX
from methods.common.torch_base import Batch, fit, n_params, person_index, to_batch
from methods.mnl.model import MNL, coefficient_table


class MixedLogit(MNL):
    panel = True

    def __init__(self, J: int, F: int, P: int, n_persons: int, *, R: int = 64, seed: int = 0):
        super().__init__(J, F, P)
        self.R = R
        self.log_s = nn.Parameter(torch.tensor([-2.0, -2.0]))
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("draws", torch.randn(n_persons, R, 2, generator=g))

    def utilities_draws(self, b: Batch) -> torch.Tensor:  # (N, R, J)
        base = self.linear_utilities(b)
        xi = self.draws[b.person]                                  # (N, R, 2)
        s = self.log_s.exp()
        beta_t = s[0] * xi[..., 0]                                 # (N, R) deviations
        beta_c = s[1] * xi[..., 1]
        t = b.X[:, :, TIME_IDX]                                    # (N, J)
        c = b.X[:, :, COST_IDX]
        return base[:, None, :] + beta_t[:, :, None] * t[:, None, :] + beta_c[:, :, None] * c[:, None, :]

    def loss(self, b: Batch) -> torch.Tensor:
        V = self.utilities_draws(b)
        lp = F.log_softmax(V, dim=-1).gather(-1, b.y[:, None, None].expand(-1, self.R, 1)).squeeze(-1)  # (N, R)
        n_persons = int(b.person.max()) + 1
        per_person = torch.zeros(n_persons, self.R).index_add(0, b.person, lp)
        ll = torch.logsumexp(per_person, dim=1) - math.log(self.R)     # (n_persons,)
        return -ll.sum() / b.n

    @torch.no_grad()
    def predict_proba(self, b: Batch):
        self.eval()
        return F.softmax(self.utilities_draws(b), dim=-1).mean(1).cpu().numpy()

    def summary(self) -> dict:
        s = self.log_s.exp().detach()
        return {"sigma_time": float(s[0]), "sigma_cost": float(s[1])}


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = MixedLogit(ds.J, ds.F, ds.P, len(pidx), R=64, seed=seed)
    info = fit(model, tr, va, lr=0.02, max_epochs=3000, patience=200, weight_decay=1e-6, seed=seed)
    extra = coefficient_table(model, ds)
    extra["random_coefficients"] = model.summary()
    return {"probs_test": model.predict_proba(te), "n_params": n_params(model), "fit": info.__dict__, "extra": extra}
