"""L-MNL (Sifringer, Lurkin & Alahi 2020): a knowledge-driven linear utility
on level-of-service attributes plus a data-driven representation term
r_j(z) learned by a neural network on the socio-demographic vector.

V_j = ASC_j + beta_j . x_j + r_j(z),   r = MLP(z) in R^J.
"""
from __future__ import annotations

import torch

from methods.common.data import ChoiceDataset
from methods.common.torch_base import Batch, fit, mlp, n_params, person_index, to_batch
from methods.mnl.model import MNL, coefficient_table


class LMNL(MNL):
    def __init__(self, J: int, F: int, P: int, hidden: int = 64):
        super().__init__(J, F, P, use_z=False)
        self.net = mlp(P, hidden, J)

    def utilities(self, b: Batch) -> torch.Tensor:
        return self.linear_utilities(b) + self.net(b.Z) * self.ref_mask


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = LMNL(ds.J, ds.F, ds.P)
    info = fit(model, tr, va, lr=2e-3, max_epochs=600, patience=40, weight_decay=1e-4, batch_size=256, seed=seed)
    return {"probs_test": model.predict_proba(te), "n_params": n_params(model), "fit": info.__dict__,
            "extra": coefficient_table(model, ds)}
