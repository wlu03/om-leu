"""Multinomial logit with alternative-specific coefficients on level of
service and alternative-specific socio-demographic shifters (the standard
Biogeme-style specification for these three datasets).

V_j = ASC_j + sum_f beta_{jf} x_{jf} + sum_p gamma_{jp} z_p,  ASC_0 = gamma_0 = 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.common.data import ChoiceDataset, TIME_IDX, COST_IDX
from methods.common.torch_base import Batch, ChoiceModel, fit, n_params, person_index, to_batch


class MNL(ChoiceModel):
    def __init__(self, J: int, F: int, P: int, *, use_z: bool = True):
        super().__init__()
        self.asc = nn.Parameter(torch.zeros(J))
        self.B = nn.Parameter(torch.zeros(J, F))
        self.G = nn.Parameter(torch.zeros(J, P))
        self.use_z = use_z
        mask = torch.ones(J)
        mask[0] = 0.0
        self.register_buffer("ref_mask", mask)

    def linear_utilities(self, b: Batch) -> torch.Tensor:
        V = self.asc * self.ref_mask + (b.X * self.B).sum(-1)
        if self.use_z:
            V = V + (b.Z @ self.G.T) * self.ref_mask
        return V

    def utilities(self, b: Batch) -> torch.Tensor:
        return self.linear_utilities(b)


def coefficient_table(model: MNL, ds: ChoiceDataset) -> dict:
    B = model.B.detach().numpy()
    out = {"asc": {a: float(v) for a, v in zip(ds.alts, model.asc.detach().numpy() * model.ref_mask.numpy())}}
    out["beta"] = {a: {f: float(B[j, k]) for k, f in enumerate(ds.alt_feature_names)} for j, a in enumerate(ds.alts)}
    vot = {}
    for j, a in enumerate(ds.alts):
        bt, bc = B[j, TIME_IDX], B[j, COST_IDX]
        if abs(bc) > 1e-6 and abs(bt) > 1e-6:
            vot[a] = float(bt / bc)  # value of time in cost units per hour
    out["value_of_time_per_hour"] = vot
    return out


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = MNL(ds.J, ds.F, ds.P)
    info = fit(model, tr, va, lr=0.02, max_epochs=4000, patience=300, weight_decay=1e-6, seed=seed)
    return {
        "probs_test": model.predict_proba(te),
        "n_params": n_params(model),
        "fit": info.__dict__,
        "extra": coefficient_table(model, ds),
    }
