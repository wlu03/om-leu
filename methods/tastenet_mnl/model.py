"""TasteNet-MNL (Han, Pereira, Ben-Akiva & Zegras 2022): a neural network
maps the person vector to taste parameters, which then enter a standard
linear-in-attributes MNL utility. Time and cost tastes are constrained
negative through a softplus so the value of time stays well defined.

beta(z), ASC(z) = TasteNet(z);  V_j = ASC_j(z) + beta(z) . x_j
VOT(z) = beta_time(z) / beta_cost(z).
(The network also outputs person-specific alternative constants, as in
the paper's "ASC + tastes" specification; the reference constant is 0.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.common.data import ChoiceDataset, TIME_IDX, COST_IDX
from methods.common.torch_base import Batch, ChoiceModel, fit, mlp, n_params, person_index, to_batch


class TasteNetMNL(ChoiceModel):
    def __init__(self, J: int, F_: int, P: int, hidden: int = 64):
        super().__init__()
        self.asc = nn.Parameter(torch.zeros(J))
        mask = torch.ones(J)
        mask[0] = 0.0
        self.register_buffer("ref_mask", mask)
        self.J, self.F_ = J, F_
        self.net = mlp(P, hidden, F_ + J)
        neg = torch.zeros(F_, dtype=torch.bool)
        neg[TIME_IDX] = True
        neg[COST_IDX] = True
        self.register_buffer("neg", neg)

    def tastes(self, b: Batch) -> torch.Tensor:  # (N, F)
        raw = self.net(b.Z)[:, : self.F_]
        return torch.where(self.neg, -F.softplus(raw), raw)

    def utilities(self, b: Batch) -> torch.Tensor:
        out = self.net(b.Z)
        raw, asc_z = out[:, : self.F_], out[:, self.F_:]
        beta = torch.where(self.neg, -F.softplus(raw), raw)
        return (self.asc + asc_z) * self.ref_mask + (b.X * beta[:, None, :]).sum(-1)


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = TasteNetMNL(ds.J, ds.F, ds.P)
    info = fit(model, tr, va, lr=2e-3, max_epochs=600, patience=40, weight_decay=1e-4, batch_size=256, seed=seed)
    with torch.no_grad():
        beta = model.tastes(te).numpy()
    vot = beta[:, TIME_IDX] / np.where(np.abs(beta[:, COST_IDX]) > 1e-6, beta[:, COST_IDX], np.nan)
    extra = {"value_of_time_per_hour_test": {"median": float(np.nanmedian(vot)), "p10": float(np.nanpercentile(vot, 10)),
                                             "p90": float(np.nanpercentile(vot, 90))}}
    return {"probs_test": model.predict_proba(te), "n_params": n_params(model), "fit": info.__dict__, "extra": extra}
