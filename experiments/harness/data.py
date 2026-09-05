"""Load an exported (dataset, seed) bundle and build the standard inputs.

Everything a model needs is on :class:`Bundle`; ``Bundle.mnl_columns`` builds
the alternative-specific MNL design (ASCs, attribute × alt, covariate × alt,
history) that the numeric residual of OM-LEU uses, so any variant can reuse
the same linear part.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
# Worktrees share the main checkout's exported tensors and results.
MAIN_REPO = Path("/Users/wesleylu/Projects/Research/structure_descent")


def data_dir() -> Path:
    p = Path(os.environ.get("OMLEU_EXPERIMENTS_DATA", "")) if os.environ.get("OMLEU_EXPERIMENTS_DATA") else None
    if p and p.exists():
        return p
    for cand in (REPO_ROOT / "experiments" / "data", MAIN_REPO / "experiments" / "data"):
        if cand.exists() and any(cand.glob("*.npz")):
            return cand
    raise FileNotFoundError("no exported tensors; run experiments/export_tensors.py in the main checkout")


def results_dir() -> Path:
    p = os.environ.get("OMLEU_EXPERIMENTS_RESULTS")
    return Path(p) if p else (MAIN_REPO / "experiments" / "results")


@dataclass
class Bundle:
    dataset: str
    seed: int
    meta: dict
    E: torch.Tensor        # (N, J, K, d) float32
    Xnum: torch.Tensor     # (N, J, F) raw units, record alt order
    Xhist: torch.Tensor    # (N, J, 2)
    z_d: torch.Tensor      # (N, p)
    Z: torch.Tensor        # (N, P) standardised
    alt_idx: torch.Tensor  # (N, J)
    y: torch.Tensor        # (N,)
    person: torch.Tensor   # (N,)
    split: torch.Tensor    # (N,) 0/1/2
    refs: Dict[str, np.ndarray] = field(default_factory=dict)

    # ---- shapes ----
    @property
    def N(self): return int(self.y.shape[0])
    @property
    def J(self): return int(self.E.shape[1])
    @property
    def K(self): return int(self.E.shape[2])
    @property
    def d(self): return int(self.E.shape[3])
    @property
    def F(self): return int(self.Xnum.shape[2])
    @property
    def P(self): return int(self.Z.shape[1])
    @property
    def n_alts(self): return len(self.meta["alts"])
    @property
    def n_persons(self): return int(self.meta["n_persons"])

    def idx(self, split: str) -> torch.Tensor:
        return torch.nonzero(self.split == {"train": 0, "val": 1, "test": 2}[split]).squeeze(1)

    # ---- standard derived inputs ----
    def alt_onehot(self) -> torch.Tensor:  # (N, J, n_alts)
        return torch.nn.functional.one_hot(self.alt_idx, self.n_alts).float()

    def xnum_std(self) -> torch.Tensor:
        """Attributes standardised per (canonical alt, feature) on train; structural zeros stay 0."""
        tr = self.idx("train")
        out = torch.zeros_like(self.Xnum)
        for a in range(self.n_alts):
            m = self.alt_idx == a                       # (N, J)
            mtr = m.clone(); mtr[self.idx("val")] = False; mtr[self.idx("test")] = False
            vals = self.Xnum[mtr]                        # (n_a, F)
            mu, sd = vals.mean(0), vals.std(0)
            sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
            keep = (vals.std(0) > 1e-6).float()
            out[m] = ((self.Xnum[m] - mu) / sd) * keep
        return out

    def mnl_columns(self, *, with_z: bool = True, with_hist: bool = True) -> tuple[torch.Tensor, List[str]]:
        """Alternative-specific MNL design (N, J, C), standardised on train."""
        oh = self.alt_onehot()                         # (N, J, A)
        A = self.n_alts
        cols, names = [], []
        for a in range(1, A):                          # ASCs, reference = alt 0
            cols.append(oh[:, :, a]); names.append(f"asc@{self.meta['alts'][a]}")
        tr_mask = torch.zeros(self.N, dtype=torch.bool); tr_mask[self.idx("train")] = True
        for f, fname in enumerate(self.meta["alt_feature_names"]):
            for a in range(A):
                col = self.Xnum[:, :, f] * oh[:, :, a]
                if col[tr_mask].std() > 1e-8:
                    cols.append(col); names.append(f"los_{fname}@{self.meta['alts'][a]}")
        if with_z:
            for p, zname in enumerate(self.meta["z_names"]):
                for a in range(1, A):
                    cols.append(self.Z[:, p][:, None] * oh[:, :, a]); names.append(f"z_{zname}@{self.meta['alts'][a]}")
        if with_hist:
            for h, hname in enumerate(self.meta["hist_names"]):
                cols.append(self.Xhist[:, :, h]); names.append(hname)
        X = torch.stack(cols, -1)                      # (N, J, C)
        flat = X[tr_mask].reshape(-1, X.shape[-1])
        mu, sd = flat.mean(0), flat.std(0)
        sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
        return (X - mu) / sd, names


def load_bundle(dataset: str, seed: int) -> Bundle:
    root = data_dir()
    z = np.load(root / f"{dataset}_seed{seed}.npz")
    meta = json.load(open(root / f"{dataset}_seed{seed}.json"))
    refs = {k: z[k] for k in z.files if k.endswith("_nll") or k.endswith("_top1")}
    t = lambda k, dt: torch.as_tensor(np.asarray(z[k]), dtype=dt)
    return Bundle(
        dataset=dataset, seed=seed, meta=meta,
        E=t("E", torch.float32), Xnum=t("Xnum", torch.float32), Xhist=t("Xhist", torch.float32),
        z_d=t("z_d", torch.float32), Z=t("Z", torch.float32), alt_idx=t("alt_idx", torch.long),
        y=t("y", torch.long), person=t("person", torch.long), split=t("split", torch.int8), refs=refs,
    )
