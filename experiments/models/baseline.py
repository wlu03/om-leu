"""In-harness reference models.

``mnl_only``     the alternative-specific MNL design of the numeric residual
                 (ASCs, attribute × alt, covariate × alt, history), fit by L-BFGS.
``hybrid_base``  that MNL, pre-fit and kept, plus an OM-LEU-style semantic
                 branch (per-sentence attribute heads, salience over K,
                 person weights over heads) behind a zero-initialised gate —
                 the current best OM-LEU variant, reproduced inside the harness.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import fit, lbfgs_prefit


def mlp(n_in, hidden, n_out, dropout=0.1):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))


class LinearPart(nn.Module):
    """β · X_mnl (standardised design), the interpretable structural utility."""

    def __init__(self, b: Bundle, *, with_z=True, with_hist=True):
        super().__init__()
        X, names = b.mnl_columns(with_z=with_z, with_hist=with_hist)
        self.register_buffer("X", X)
        self.names = names
        self.beta = nn.Parameter(torch.zeros(X.shape[-1]))

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return (self.X[idx] * self.beta).sum(-1)

    def prefit(self, b: Bundle, l2: float = 1e-4) -> float:
        tr = b.idx("train")

        def loss():
            return F.cross_entropy(self.forward(b, tr), b.y[tr]) + l2 * (self.beta ** 2).sum()
        return lbfgs_prefit(loss, [self.beta])


class SemanticBranch(nn.Module):
    """OM-LEU-style: heads over each sentence, salience over K, person weights over M heads."""

    def __init__(self, b: Bundle, M: int = 5, hidden: int = 128, w_hidden: int = 32, s_hidden: int = 64):
        super().__init__()
        self.heads = mlp(b.d, hidden, M)
        self.salience = mlp(b.d, s_hidden, 1)
        self.weights = mlp(b.z_d.shape[1], w_hidden, M, dropout=0.0)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        E = b.E[idx]                                    # (n, J, K, d)
        A = self.heads(E)                               # (n, J, K, M)
        S = torch.softmax(self.salience(E).squeeze(-1), dim=2)   # (n, J, K)
        U = (A * S[..., None]).sum(2)                   # (n, J, M)
        w = torch.softmax(self.weights(b.z_d[idx]), dim=-1)      # (n, M)
        return (U * w[:, None, :]).sum(-1)              # (n, J)


class Hybrid(nn.Module):
    def __init__(self, b: Bundle, *, gate_init: float = 0.0, **sem_kw):
        super().__init__()
        self.lin = LinearPart(b)
        self.sem = SemanticBranch(b, **sem_kw)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, b, idx):
        return self.lin(b, idx) + self.gate * self.sem(b, idx)


def build_mnl_only(b: Bundle, seed: int):
    model = LinearPart(b)

    def run(model, b, seed):
        ce = model.prefit(b)
        return {"fit": {"train_ce": ce}, "extra": {"n_columns": len(model.names)}}
    return model, run


def build_hybrid_base(b: Bundle, seed: int):
    model = Hybrid(b)

    def run(model, b, seed):
        ce = model.lin.prefit(b)
        sem_params = list(model.sem.parameters()) + [model.gate]
        fr = fit(model, b, params=sem_params, lr=1e-3, max_epochs=60, patience=8, batch_size=256, seed=seed)
        return {"fit": {**fr.__dict__, "stage1_train_ce": ce}, "extra": {"gate": float(model.gate)}}
    return model, run
