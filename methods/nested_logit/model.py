"""Nested logit (Ben-Akiva & Lerman form) on the MNL specification of
``methods/mnl``. Nests per dataset are in ``methods.common.data.NESTS``:
Swissmetro {train, Swissmetro} vs {car}; Optima {car} vs {PT, soft};
LPMC {walk, cycle} vs {PT, drive}.

P(j) = P(k(j)) * P(j | k),  P(j|k) = exp(mu_k V_j) / sum_{i in k} exp(mu_k V_i),
P(k) = exp(L_k / mu_k) / sum_l exp(L_l / mu_l),  L_k = log sum_{i in k} exp(mu_k V_i),
mu_k = 1 + softplus(theta_k) >= 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common.data import ChoiceDataset
from methods.common.torch_base import Batch, fit, n_params, person_index, to_batch
from methods.mnl.model import MNL, coefficient_table


class NestedLogit(MNL):
    def __init__(self, J: int, F: int, P: int, nests: dict):
        super().__init__(J, F, P)
        self.nest_names = list(nests)
        alt_to_nest = torch.zeros(J, dtype=torch.long)
        for k, (name, members) in enumerate(nests.items()):
            for j in members:
                alt_to_nest[j] = k
        self.register_buffer("alt_to_nest", alt_to_nest)
        self.theta = nn.Parameter(torch.full((len(nests),), -2.0))  # mu ~ 1.13 at init

    def mu(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.theta)

    def log_probs(self, b: Batch) -> torch.Tensor:
        V = self.linear_utilities(b)                          # (N, J)
        mu = self.mu()                                        # (K,)
        mu_j = mu[self.alt_to_nest]                           # (J,)
        scaled = mu_j * V                                     # (N, J)
        K = mu.shape[0]
        onehot = F.one_hot(self.alt_to_nest, K).float()       # (J, K)
        masked = scaled[:, :, None] + torch.log(onehot)[None]  # -inf outside nest
        L = torch.logsumexp(masked, dim=1)                    # (N, K)
        log_p_within = scaled - L[:, self.alt_to_nest]        # (N, J)
        log_p_nest = F.log_softmax(L / mu, dim=1)             # (N, K)
        return log_p_within + log_p_nest[:, self.alt_to_nest]

    def log_prob(self, b: Batch) -> torch.Tensor:
        return self.log_probs(b).gather(1, b.y[:, None]).squeeze(1)

    @torch.no_grad()
    def predict_proba(self, b: Batch):
        self.eval()
        return self.log_probs(b).exp().cpu().numpy()


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = NestedLogit(ds.J, ds.F, ds.P, ds.nests)
    info = fit(model, tr, va, lr=0.02, max_epochs=4000, patience=300, weight_decay=1e-6, seed=seed)
    extra = coefficient_table(model, ds)
    extra["nest_scale_mu"] = {n: float(v) for n, v in zip(model.nest_names, model.mu().detach().numpy())}
    return {"probs_test": model.predict_proba(te), "n_params": n_params(model), "fit": info.__dict__, "extra": extra}
