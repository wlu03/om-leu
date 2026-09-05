"""LightGBM child process for the boosted-residual variants (experiments/models/boost.py).

Runs in its own interpreter because LightGBM and torch cannot share a process
on this machine (duplicate libomp). Imports only numpy + lightgbm.

    python _lgb_child.py <in.npz> <out.npz>

``in.npz`` holds ``cfg`` (json string) and, per split s in {tr, va, te}:
  multiclass mode:  X_<s> (n, D), y_<s> (n,), init_<s> (n, A)
  rum mode:         X<a>_<s> (n, D_a) for each alternative a, y_<s>, init_<s>, mono<a> (D_a,)
``out.npz`` holds raw utilities U_<s> (n, A) (init included), best_round, n_trees,
and per-feature split-gain importances.
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np


def _softmax(U):
    U = U - U.max(1, keepdims=True)
    e = np.exp(U)
    return e / e.sum(1, keepdims=True)


def _nll(U, y):
    P = _softmax(U)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean())


def _base_params(cfg):
    return {
        "learning_rate": cfg["lr"], "num_leaves": cfg["num_leaves"], "min_data_in_leaf": cfg["min_data"],
        "feature_fraction": cfg.get("feature_fraction", 0.8), "bagging_fraction": cfg.get("bagging_fraction", 0.8),
        "bagging_freq": 1, "lambda_l2": cfg.get("lambda_l2", 1.0), "max_bin": cfg.get("max_bin", 255),
        "verbose": -1, "seed": cfg["seed"], "num_threads": cfg.get("num_threads", 4),
    }


def run_multiclass(d, cfg):
    A = int(cfg["A"])
    params = {**_base_params(cfg), "objective": "multiclass", "num_class": A}
    use_init = bool(cfg.get("use_init", True))
    kw = lambda s: {"init_score": d[f"init_{s}"]} if use_init else {}
    dtr = lgb.Dataset(d["X_tr"], label=d["y_tr"], **kw("tr"), params={"verbose": -1})
    dva = lgb.Dataset(d["X_va"], label=d["y_va"], reference=dtr, **kw("va"), params={"verbose": -1})
    bst = lgb.train(params, dtr, num_boost_round=cfg["max_rounds"], valid_sets=[dva],
                    callbacks=[lgb.early_stopping(cfg["patience"], verbose=False)])
    best = int(bst.best_iteration)
    out = {}
    for s in ("tr", "va", "te"):
        raw = bst.predict(d[f"X_{s}"], raw_score=True, num_iteration=best).reshape(len(d[f"y_{s}"]), A)
        out[f"U_{s}"] = raw + (d[f"init_{s}"] if use_init else 0.0)
    out["best_round"] = best
    out["n_trees"] = int(bst.num_trees())
    out["importance"] = bst.feature_importance("gain")
    out["best_val_nll"] = _nll(out["U_va"], d["y_va"])
    return out


def run_rum(d, cfg):
    """One booster per alternative on that alternative's own feature block, trained jointly
    under the softmax cross-entropy (RUMBoost); utilities start at the init score."""
    A = int(cfg["A"])
    use_init = bool(cfg.get("use_init", True))
    init = {s: (d[f"init_{s}"].astype(np.float64) if use_init else np.zeros_like(d[f"init_{s}"], dtype=np.float64))
            for s in ("tr", "va", "te")}
    boosters = []
    for a in range(A):
        n_feat = d[f"X{a}_tr"].shape[1]
        p = {**_base_params(cfg), "objective": "regression"}
        if cfg.get("monotone", True):
            p["monotone_constraints"] = [int(v) for v in d[f"mono{a}"]]
            p["monotone_constraints_method"] = "advanced"
        if cfg.get("additive", True):
            p["interaction_constraints"] = [[i] for i in range(n_feat)]
        dtr = lgb.Dataset(d[f"X{a}_tr"], label=np.zeros(len(d["y_tr"])), free_raw_data=False, params={"verbose": -1})
        boosters.append(lgb.Booster(p, dtr))
    y_tr, y_va = d["y_tr"], d["y_va"]
    Y = np.eye(A)[y_tr]
    U_tr, U_va = init["tr"].copy(), init["va"].copy()
    best, best_round, bad, curve = np.inf, 0, 0, []
    r = 0
    for r in range(1, cfg["max_rounds"] + 1):
        for a in range(A):
            P = _softmax(U_tr)
            g = (P[:, a] - Y[:, a]).astype(np.float64)
            h = (P[:, a] * (1 - P[:, a])).astype(np.float64)
            boosters[a].update(fobj=lambda preds, data, g=g, h=h: (g, h))
            k = boosters[a].num_trees() - 1
            U_tr[:, a] += boosters[a].predict(d[f"X{a}_tr"], raw_score=True, start_iteration=k, num_iteration=1)
            U_va[:, a] += boosters[a].predict(d[f"X{a}_va"], raw_score=True, start_iteration=k, num_iteration=1)
        v = _nll(U_va, y_va)
        curve.append(v)
        if v < best - 1e-6:
            best, best_round, bad = v, r, 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    out = {}
    for s in ("tr", "va", "te"):
        U = init[s].copy()
        if best_round > 0:
            for a in range(A):
                U[:, a] += boosters[a].predict(d[f"X{a}_{s}"], raw_score=True, num_iteration=best_round)
        out[f"U_{s}"] = U
    out["best_round"] = best_round
    out["n_trees"] = int(sum(min(b.num_trees(), best_round) for b in boosters))
    out["best_val_nll"] = best
    out["rounds_run"] = r
    for a in range(A):
        out[f"importance{a}"] = boosters[a].feature_importance("gain", iteration=best_round) if best_round > 0 \
            else np.zeros(d[f"X{a}_tr"].shape[1])
    return out


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    z = np.load(inp)
    d = {k: z[k] for k in z.files}
    cfg = json.loads(str(d.pop("cfg")))
    out = run_rum(d, cfg) if cfg["mode"] == "rum" else run_multiclass(d, cfg)
    np.savez(outp, **out)


if __name__ == "__main__":
    main()
