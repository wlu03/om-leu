"""RUMBoost (Salvadé & Hillel 2024): gradient-boosted random utility model.
One LightGBM ensemble per alternative forms that alternative's utility
from its own attributes plus the person vector; the ensembles are trained
jointly under the multinomial-logit cross-entropy (softmax over the J
utilities), so each boosting step uses the RUM gradient
g_j = P_j - 1[y = j], h_j = P_j (1 - P_j). Utilities are additive in the
features (each tree splits on a single feature) and monotone decreasing in
time and cost, which keeps the model interpretable as a RUM.

Implemented directly on LightGBM because the ``rumboost`` PyPI package
pins a Biogeme build that does not import under NumPy 2 in this venv.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from methods.common.data import ChoiceDataset, TIME_IDX, COST_IDX


def _alt_matrix(ds: ChoiceDataset, split: str, j: int, keep: np.ndarray) -> np.ndarray:
    s = ds.splits[split]
    return np.concatenate([s.X[:, j, :][:, keep], s.Z], axis=1).astype(np.float32)


def _softmax(U: np.ndarray) -> np.ndarray:
    U = U - U.max(1, keepdims=True)
    e = np.exp(U)
    return e / e.sum(1, keepdims=True)


def _nll(U: np.ndarray, y: np.ndarray) -> float:
    P = _softmax(U)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def run(ds: ChoiceDataset, seed: int, *, lr: float = 0.1, max_rounds: int = 1500, patience: int = 50,
        num_leaves: int = 31, min_data: int = 20, additive: bool = True, monotone: bool = True, **kw) -> dict:
    t0 = time.time()
    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]
    J, F = ds.J, ds.F
    boosters, mats = [], {}
    for j in range(J):
        keep = np.where(tr.X[:, j, :].std(0) > 1e-8)[0]          # drop structurally-zero attributes
        n_feat = len(keep) + ds.P
        mono = [0] * n_feat
        if monotone:
            for k, f in enumerate(keep):
                if f in (TIME_IDX, COST_IDX):
                    mono[k] = -1
        params = {
            "objective": "regression", "learning_rate": lr, "num_leaves": num_leaves,
            "min_data_in_leaf": min_data, "verbose": -1, "seed": seed, "num_threads": 4,
            "monotone_constraints": mono, "monotone_constraints_method": "advanced",
        }
        if additive:
            params["interaction_constraints"] = [[i] for i in range(n_feat)]
        mats[j] = {s: _alt_matrix(ds, s, j, keep) for s in ("train", "val", "test")}
        dtrain = lgb.Dataset(mats[j]["train"], label=np.zeros(tr.n), free_raw_data=False, params={"verbose": -1})
        boosters.append(lgb.Booster(params, dtrain))
    Y = np.eye(J)[tr.y]
    U_tr = np.zeros((tr.n, J)); U_va = np.zeros((va.n, J))
    best, best_round, bad = np.inf, 0, 0
    for r in range(1, max_rounds + 1):
        for j in range(J):
            P = _softmax(U_tr)
            grad = (P[:, j] - Y[:, j]).astype(np.float64)
            hess = (P[:, j] * (1 - P[:, j])).astype(np.float64)
            boosters[j].update(fobj=lambda preds, data, g=grad, h=hess: (g, h))
            U_tr[:, j] = boosters[j].predict(mats[j]["train"], raw_score=True)
            U_va[:, j] = boosters[j].predict(mats[j]["val"], raw_score=True)
        v = _nll(U_va, va.y)
        if v < best - 1e-6:
            best, best_round, bad = v, r, 0
        else:
            bad += 1
        if bad >= patience:
            break
    U_te = np.stack([boosters[j].predict(mats[j]["test"], raw_score=True, num_iteration=best_round) for j in range(J)], 1)
    n_leaves = int(sum(b.num_trees() for b in boosters)) * num_leaves
    return {
        "probs_test": _softmax(U_te), "n_params": n_leaves,
        "fit": {"best_round": best_round, "best_val_nll": best, "rounds_run": r, "seconds": time.time() - t0},
        "extra": {"additive": additive, "monotone_time_cost": monotone},
    }
