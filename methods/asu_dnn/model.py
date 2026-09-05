"""ASU-DNN (Wang, Wang & Zhao 2020): alternative-specific utility deep
neural network. Each alternative has its own sub-network that sees only
that alternative's attributes and the individual's characteristics, so
cross-alternative attribute leakage into a utility is ruled out (the
DNN analogue of a random-utility specification).

V_j = f_j([x_j, z]),  f_j separate MLPs.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from methods.common.data import ChoiceDataset
from methods.common.torch_base import Batch, ChoiceModel, fit, mlp, n_params, person_index, to_batch


class ASUDNN(ChoiceModel):
    def __init__(self, J: int, F: int, P: int, hidden: int = 64):
        super().__init__()
        self.nets = nn.ModuleList([mlp(F + P, hidden, 1) for _ in range(J)])

    def utilities(self, b: Batch) -> torch.Tensor:
        return torch.cat([net(torch.cat([b.Xs[:, j, :], b.Z], 1)) for j, net in enumerate(self.nets)], 1)


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    pidx = person_index(ds)
    tr, va, te = (to_batch(ds, s, pidx) for s in ("train", "val", "test"))
    model = ASUDNN(ds.J, ds.F, ds.P)
    info = fit(model, tr, va, lr=1e-3, max_epochs=600, patience=40, weight_decay=1e-4, batch_size=256, seed=seed)
    return {"probs_test": model.predict_proba(te), "n_params": n_params(model), "fit": info.__dict__, "extra": {}}
