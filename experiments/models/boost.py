"""Boosted residual on top of the interpretable hybrid (stacked boosting).

Stage 1  fit the hybrid exactly as ``hybrid_base`` (LinearPart by L-BFGS,
         semantic branch behind a zero-init gate) and take its logits.
Stage 2  LightGBM in a child process (torch and LightGBM cannot share a
         process here) with the hybrid logits as ``init_score`` so the trees
         only learn what the structural utility misses.  Two shapes:

  rum         one booster per alternative on that alternative's own block
              (standardised attributes, person covariates, history, optional
              semantic features); monotone −1 on time and cost; optionally
              one feature per tree (additive, RUMBoost).  Keeps the RUM form.
  multiclass  plain multiclass booster on the flattened features (ceiling).

Semantic features (``sem``): ``pca16`` = PCA-16 of the mean-over-K sentence
embedding fit on train; ``heads`` = per-alternative head scores of a
semantic branch trained on its own (the hybrid's branch is untrained when the
gate stays at 0).  ``init``: ``hybrid`` (in-sample logits), ``oof`` (5-fold
out-of-fold LinearPart logits for train rows, full-train fit for val/test),
``lean``/``lean_oof`` (LinearPart without covariate × alt columns) or ``none``
(plain GBDT inside the harness, sanity check).  ``init_scale`` shrinks the
offset (tau); ``init_as_feature`` puts the structural utility into each
alternative's feature block (monotone +1) instead of the offset.

Findings (3 seeds, see experiments/results/summary.md): the raw offset
(tau = 1) is useless because the linear part overfits train; tau = 0.5 with a
RUM-shaped booster (``boost_v2``) beats hybrid_base significantly on all three
datasets and the methods/ GBDT on all three; semantic features help only on
LPMC (not significant), hurt on Optima.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import fit, lbfgs_prefit
from experiments.models.baseline import Hybrid, LinearPart, SemanticBranch

CHILD = Path(__file__).resolve().parent / "_lgb_child.py"


# ----------------------------------------------------------------------------- stage 1
def fit_hybrid(b: Bundle, seed: int):
    model = Hybrid(b)
    ce = model.lin.prefit(b)
    fr = fit(model, b, params=list(model.sem.parameters()) + [model.gate], lr=1e-3, max_epochs=60, patience=8,
             batch_size=256, seed=seed)
    return model, {"stage1_train_ce": ce, "hybrid_best_epoch": fr.best_epoch, "hybrid_val_nll": fr.best_val_nll,
                   "gate": float(model.gate)}


@torch.no_grad()
def all_logits(model: nn.Module, b: Bundle) -> torch.Tensor:
    idx = torch.arange(b.N)
    model.eval()
    return torch.cat([model(b, idx[s:s + 2048]) for s in range(0, b.N, 2048)], 0)


def oof_linear_logits(b: Bundle, seed: int, folds: int = 5, with_z: bool = True) -> torch.Tensor:
    """LinearPart logits: out-of-fold (by person) on train rows, full-train fit elsewhere."""
    full = LinearPart(b, with_z=with_z)
    full.prefit(b)
    out = all_logits(full, b).clone()
    tr = b.idx("train")
    persons = b.person[tr].numpy()
    uniq = np.unique(persons)
    rng = np.random.default_rng(seed)
    fold_of_person = dict(zip(uniq, rng.integers(0, folds, len(uniq))))
    fold = np.array([fold_of_person[p] for p in persons])
    for k in range(folds):
        fit_idx, hold_idx = tr[torch.as_tensor(fold != k)], tr[torch.as_tensor(fold == k)]
        m = LinearPart(b, with_z=with_z)

        def loss(m=m, fit_idx=fit_idx):
            return F.cross_entropy(m(b, fit_idx), b.y[fit_idx]) + 1e-4 * (m.beta ** 2).sum()
        lbfgs_prefit(loss, [m.beta])
        with torch.no_grad():
            out[hold_idx] = m(b, hold_idx)
    return out


class SemOnly(nn.Module):
    def __init__(self, b: Bundle):
        super().__init__()
        self.sem = SemanticBranch(b)
        self.asc = nn.Parameter(torch.zeros(b.n_alts))

    def forward(self, b, idx):
        return self.sem(b, idx) + self.asc[b.alt_idx[idx]]


def head_features(b: Bundle, seed: int) -> tuple[torch.Tensor, List[str], dict]:
    """Train the semantic branch alone; return (N, J, M+1) head scores + semantic utility."""
    m = SemOnly(b)
    fr = fit(m, b, lr=1e-3, max_epochs=60, patience=8, batch_size=256, seed=seed)
    m.eval()
    return _head_scores(m, b, fr)


@torch.no_grad()
def _head_scores(m: "SemOnly", b: Bundle, fr) -> tuple[torch.Tensor, List[str], dict]:
    feats = []
    for s in range(0, b.N, 2048):
        idx = torch.arange(s, min(s + 2048, b.N))
        E = b.E[idx]
        A = m.sem.heads(E)
        S = torch.softmax(m.sem.salience(E).squeeze(-1), dim=2)
        U = (A * S[..., None]).sum(2)                                    # (n, J, M)
        w = torch.softmax(m.sem.weights(b.z_d[idx]), dim=-1)
        feats.append(torch.cat([U, (U * w[:, None, :]).sum(-1, keepdim=True)], -1))
    X = torch.cat(feats, 0)
    names = [f"sem_head{k}" for k in range(X.shape[-1] - 1)] + ["sem_util"]
    return X, names, {"sem_only_val_nll": fr.best_val_nll, "sem_only_best_epoch": fr.best_epoch}


def pca_features(b: Bundle, n_comp: int = 16) -> tuple[torch.Tensor, List[str]]:
    """PCA of the mean-over-K sentence embedding, fit on train rows (all slots)."""
    Em = b.E.mean(2)                                                     # (N, J, d)
    tr = b.idx("train")
    flat = Em[tr].reshape(-1, b.d).double()
    mu = flat.mean(0)
    cov = (flat - mu).T @ (flat - mu) / (flat.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)
    W = evecs[:, -n_comp:].flip(1)                                       # (d, n_comp), leading first
    proj = ((Em.double() - mu) @ W).float()
    sd = proj[tr].reshape(-1, n_comp).std(0).clamp_min(1e-6)
    return proj / sd, [f"sem_pc{k}" for k in range(n_comp)]


# ----------------------------------------------------------------------------- features
def _canon(b: Bundle, X: torch.Tensor) -> torch.Tensor:
    """(N, J, ...) in record slot order -> canonical alternative order."""
    pos = b.alt_idx.argsort(1)                                          # pos[i, a] = slot of alt a
    idx = pos.reshape(pos.shape + (1,) * (X.dim() - 2)).expand(pos.shape + X.shape[2:])
    return X.gather(1, idx)


def build_blocks(b: Bundle, sem: Optional[str], seed: int, extra: Optional[torch.Tensor] = None):
    """Per canonical alternative a: (feature matrix (N, D_a), names, monotone vector).
    ``extra`` (N, A) adds a per-alternative column (the structural utility as a feature, monotone +1)."""
    xs = _canon(b, b.xnum_std())                                        # (N, A, F)
    hist = _canon(b, b.Xhist)                                           # (N, A, 2)
    info = {}
    if sem == "pca16":
        S, snames = pca_features(b, 16); S = _canon(b, S)
    elif sem == "heads":
        S, snames, info = head_features(b, seed); S = _canon(b, S)
    else:
        S, snames = None, []
    tr = b.idx("train")
    blocks = []
    for a in range(b.n_alts):
        cols, names, mono = [], [], []
        for f, fname in enumerate(b.meta["alt_feature_names"]):
            col = xs[:, a, f]
            if col[tr].std() > 1e-8:
                cols.append(col); names.append(f"los_{fname}")
                mono.append(-1 if f in (b.meta["time_idx"], b.meta["cost_idx"]) else 0)
        for p, zname in enumerate(b.meta["z_names"]):
            cols.append(b.Z[:, p]); names.append(f"z_{zname}"); mono.append(0)
        for h, hname in enumerate(b.meta["hist_names"]):
            cols.append(hist[:, a, h]); names.append(f"hist_{hname}"); mono.append(0)
        if S is not None:
            for k, sname in enumerate(snames):
                cols.append(S[:, a, k]); names.append(sname); mono.append(0)
        if extra is not None:
            cols.append(extra[:, a]); names.append("v_struct"); mono.append(1)
        blocks.append((torch.stack(cols, 1).numpy().astype(np.float32), names, np.array(mono, dtype=np.int64)))
    return blocks, info


# ----------------------------------------------------------------------------- stage 2
def run_child(payload: dict, cfg: dict) -> dict:
    with tempfile.TemporaryDirectory() as td:
        inp, outp = os.path.join(td, "in.npz"), os.path.join(td, "out.npz")
        np.savez(inp, cfg=np.array(json.dumps(cfg)), **payload)
        env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
        r = subprocess.run([sys.executable, str(CHILD), inp, outp], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"lgb child failed (rc={r.returncode}):\n{r.stderr[-4000:]}")
        z = np.load(outp)
        return {k: z[k] for k in z.files}


class BoostedModel(nn.Module):
    """Holds the stage-1 hybrid (for parameter counting) and the final logits."""

    def __init__(self, b: Bundle, hybrid: nn.Module):
        super().__init__()
        self.hybrid = hybrid
        self.register_buffer("logits", torch.zeros(b.N, b.J))

    def forward(self, b, idx):
        return self.logits[idx]


DEFAULT = dict(mode="rum", additive=True, monotone=True, sem=None, init="hybrid",
               lr=0.03, num_leaves=8, min_data=20, lambda_l2=1.0, feature_fraction=0.8, bagging_fraction=0.8,
               patience=100, max_rounds=3000, num_threads=4)


def _build(b: Bundle, seed: int, cfg: dict):
    cfg = {**DEFAULT, **cfg}
    hybrid, info1 = fit_hybrid(b, seed)
    model = BoostedModel(b, hybrid)

    def run(model, b, seed):
        t0 = time.time()
        if cfg["init"] == "hybrid":
            init = all_logits(hybrid, b)
        elif cfg["init"] == "oof":
            init = oof_linear_logits(b, seed)
        elif cfg["init"] == "lean":                                      # ASCs + attributes + history only
            lean = LinearPart(b, with_z=False); lean.prefit(b); init = all_logits(lean, b)
        elif cfg["init"] == "lean_oof":
            init = oof_linear_logits(b, seed, with_z=False)
        else:
            init = torch.zeros(b.N, b.J)
        init = init * float(cfg.get("init_scale", 1.0))
        init_c = _canon(b, init).numpy().astype(np.float64)              # (N, A)
        y_c = b.alt_idx[torch.arange(b.N), b.y].numpy()                  # canonical chosen alt
        extra_col = None
        if cfg.get("init_as_feature", False):                            # stacking in feature space instead of offset
            extra_col = torch.as_tensor(init_c, dtype=torch.float32)
            init_c = np.zeros_like(init_c)
        blocks, info_sem = build_blocks(b, cfg["sem"], seed, extra=extra_col)
        splits = {s: b.idx(s).numpy() for s in ("train", "val", "test")}
        tag = {"train": "tr", "val": "va", "test": "te"}
        payload = {}
        for s, ix in splits.items():
            payload[f"y_{tag[s]}"] = y_c[ix]
            payload[f"init_{tag[s]}"] = init_c[ix]
        ccfg = {"mode": cfg["mode"], "A": b.n_alts, "seed": seed,
                "use_init": cfg["init"] != "none" and not cfg.get("init_as_feature", False),
                **{k: cfg[k] for k in ("lr", "num_leaves", "min_data", "lambda_l2", "feature_fraction",
                                       "bagging_fraction", "patience", "max_rounds", "num_threads",
                                       "additive", "monotone")}}
        if cfg["mode"] == "rum":
            for a, (X, names, mono) in enumerate(blocks):
                payload[f"mono{a}"] = mono
                for s, ix in splits.items():
                    payload[f"X{a}_{tag[s]}"] = X[ix]
            feat_names = [[f"{b.meta['alts'][a]}:{n}" for n in names] for a, (_, names, _) in enumerate(blocks)]
        else:
            Xall, names = [], []
            seen_z = False
            for a, (X, nm, _) in enumerate(blocks):
                keep = [i for i, n in enumerate(nm) if not (n.startswith("z_") and seen_z)]
                seen_z = True
                Xall.append(X[:, keep]); names += [f"{b.meta['alts'][a]}:{nm[i]}" if not nm[i].startswith("z_") else nm[i] for i in keep]
            Xall = np.concatenate(Xall, 1)
            for s, ix in splits.items():
                payload[f"X_{tag[s]}"] = Xall[ix]
            feat_names = [names]
        out = run_child(payload, ccfg)
        U = torch.zeros(b.N, b.n_alts, dtype=torch.float64)
        for s, ix in splits.items():
            U[torch.as_tensor(ix)] = torch.as_tensor(out[f"U_{tag[s]}"], dtype=torch.float64)
        model.logits.copy_(U.float().gather(1, b.alt_idx))               # back to record slot order
        # importances (top 8 per booster) for the report
        imps = {}
        if cfg["mode"] == "rum":
            for a in range(b.n_alts):
                imp = out[f"importance{a}"]; order = np.argsort(-imp)[:8]
                tot = imp.sum() or 1.0
                imps[b.meta["alts"][a]] = {feat_names[a][i]: round(float(imp[i] / tot), 3) for i in order if imp[i] > 0}
        else:
            imp = out["importance"]; order = np.argsort(-imp)[:12]; tot = imp.sum() or 1.0
            imps["all"] = {feat_names[0][i]: round(float(imp[i] / tot), 3) for i in order if imp[i] > 0}
        hyb_val = float(F.cross_entropy(init[b.idx("val")], b.y[b.idx("val")])) if cfg["init"] != "none" else None
        return {"fit": {"best_epoch": int(out["best_round"]), "best_round": int(out["best_round"]),
                        "n_trees": int(out["n_trees"]), "best_val_nll": float(out["best_val_nll"]),
                        "init_val_nll": hyb_val, "seconds": time.time() - t0, **info1, **info_sem},
                "extra": {"cfg": cfg, "n_features": [X.shape[1] for X, _, _ in blocks], "importance": imps}}
    return model, run


# ----------------------------------------------------------------------------- registry entries
RUM = dict(mode="rum", additive=False, monotone=True, num_leaves=15)     # interactions within an alternative's block
MC = dict(mode="multiclass", num_leaves=15)
# Registered (best two shapes, each with/without semantic features):
#   boost_v2  residual RUM: hybrid logits as offset shrunk by 0.5 — significant on all three datasets
#   boost_v3  residual multiclass, same offset (unconstrained ceiling) — good on Optima/LPMC, not Swissmetro
# The remaining configs stay here for reproducibility of the iteration (results under experiments/results/):
#   boost_v1 stacked RUM (hybrid utility as monotone feature), boost_v4/v5 boosters from scratch,
#   boost_add the original additive RUMBoost spec, *_heads head-score semantic features.
# tau = 1 (the raw hybrid offset) leaves nothing to learn on train (the linear part overfits train:
# CE 0.37 vs val 0.49 on Optima) and the residual stops after a handful of rounds.
VARIANTS: Dict[str, dict] = {
    "boost_v2":       dict(**RUM, init="hybrid", init_scale=0.5, sem=None),
    "boost_v2_sem":   dict(**RUM, init="hybrid", init_scale=0.5, sem="pca16"),
    "boost_v3":       dict(**MC, init="hybrid", init_scale=0.5, sem=None),
    "boost_v3_sem":   dict(**MC, init="hybrid", init_scale=0.5, sem="pca16"),
    # --- not registered ---
    "boost_v1":       dict(**RUM, init="hybrid", init_as_feature=True, sem=None),
    "boost_v1_sem":   dict(**RUM, init="hybrid", init_as_feature=True, sem="pca16"),
    "boost_v1_heads": dict(**RUM, init="hybrid", init_as_feature=True, sem="heads"),
    "boost_v3_heads": dict(**MC, init="hybrid", init_scale=0.5, sem="heads"),
    "boost_v4":       dict(**RUM, init="none", sem=None),
    "boost_v5":       dict(**MC, init="none", sem=None),
    "boost_add":      dict(mode="rum", additive=True, monotone=True, num_leaves=8, init="hybrid", init_scale=0.5, sem=None),
}

for _name, _cfg in VARIANTS.items():
    globals()[f"build_{_name}"] = partial(_build, cfg=_cfg)
