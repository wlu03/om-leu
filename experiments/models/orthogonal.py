"""Make the sentence contribution complementary to the numbers by construction (OM-LEU 2, stage 3).

Two devices, usable separately or together (``Config.erase`` / ``Config.resid_nu`` in omleu2.py):

1. LEACE-style concept erasure (Belrose et al. 2023, regression form).  Before the semantic
   members see the outcome sentences, every sentence embedding is residualised on the concept
   vector c_ij = [1, standardised numeric attributes of the alternative (``Bundle.xnum_std``),
   person covariates Z_i], with a separate least-squares map per (sentence slot k, canonical
   alternative a) fitted on the training rows:

       E'_ijk = E_ijk - c_ij . W_{k, a(ij)}      W = argmin_W || E - c W ||^2 on train, a = alt of (i, j)

   Fitting per alternative removes the per-alternative mean (alternative identity) and every
   within-alternative direction linearly predictable from numbers and covariates, so on the
   training rows Cov(E', [x, alt one-hot, Z]) = 0 exactly: no linear reader of E' recovers any of
   them.  ``_ErasedE`` is a lazy view (works under ``_ShuffledE`` / ``_SlotE``); the shuffled control
   of an erased variant shuffles the erased embeddings (erase first, then shuffle, so the control
   cannot leak numbers through the subtracted term).

   ``probe_r2`` reports the ridge-probe R² of time, cost, alternative identity and Z from the
   concatenated K slot embeddings, before and after erasure (train and test rows).

2. Residual training.  ``ResidualMember`` wraps a ``PrefBranch`` and adds nu . L_ij to its
   utilities, where L is the stage-1 structural log-probability (cross-fitted on training rows,
   full-train fit on val / test) and nu in {0.3, 1} a fixed shrinkage:

       log p_m(j | E, z, x) = log softmax_j ( V_m(E, z) + nu L(x, z) )

   The member is trained (choice cross-entropy + the InfoNCE probe, as before) with L fixed, so
   V_m can only learn what the structural part misses; the probability-level mixture with the
   boosted structural part is unchanged at inference.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.models.pref import PrefBranch, _log_mean_exp


# ----------------------------------------------------------------------------- concepts
def concept_matrix(b: Bundle) -> torch.Tensor:
    """c_ij = [1, xnum_std_ij, Z_i]  (N, J, 1 + F + P)."""
    X = b.xnum_std()
    Z = b.Z[:, None, :].expand(-1, b.J, -1)
    return torch.cat([torch.ones(b.N, b.J, 1), X, Z], -1)


class _ErasedE:
    """Lazy E view: E'_ijk = E_ijk - c_ij . W[k, alt_ij]; supports ``E[idx]`` and ``E[i, j]`` indexing."""

    def __init__(self, E: torch.Tensor, c: torch.Tensor, W: torch.Tensor, alt_idx: torch.Tensor):
        self._E, self._c, self._W, self._alt = E, c, W, alt_idx          # W: (K, A, C, d)
        self.shape = E.shape

    def _correct(self, c: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # c: (..., C), a: (...) -> (..., K, d); corrections for every alternative, then gather the row's own
        corr = torch.einsum("...c,kacd->...kad", c, self._W)            # (..., K, A, d)
        ix = a[..., None, None, None].expand(*a.shape, self._W.shape[0], 1, self._W.shape[-1])
        return corr.gather(-2, ix).squeeze(-2)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            i, j = idx
            return self._E[i, j] - self._correct(self._c[i, j], self._alt[i, j])
        return self._E[idx] - self._correct(self._c[idx], self._alt[idx])


def fit_erasure(b: Bundle) -> Tuple[torch.Tensor, torch.Tensor]:
    """Least-squares maps W[k, a] from c to E_k on the training rows of alternative a."""
    c = concept_matrix(b)
    tr = torch.zeros(b.N, dtype=torch.bool); tr[b.idx("train")] = True
    W = torch.zeros(b.K, b.n_alts, c.shape[-1], b.d, dtype=torch.float64)
    for a in range(b.n_alts):
        pos = torch.nonzero((b.alt_idx == a) & tr[:, None], as_tuple=False)
        Ca = c[pos[:, 0], pos[:, 1]].double()                           # (n_a, C)
        for k in range(b.K):
            Ek = b.E[pos[:, 0], pos[:, 1], k].double()                  # (n_a, d)
            W[k, a] = torch.linalg.lstsq(Ca, Ek, driver="gelsd").solution
    return c, W.float()


def erased_view(b: Bundle) -> Bundle:
    c, W = fit_erasure(b)
    return dataclasses.replace(b, E=_ErasedE(b.E, c, W, b.alt_idx))


# ----------------------------------------------------------------------------- linear probe
def _flat_E(E, b: Bundle, rows: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Concatenated K-slot embeddings of every (event, slot) pair in ``rows`` (n·J, K·d)."""
    out = []
    for s in range(0, len(rows), chunk):
        sel = rows[s:s + chunk]
        out.append(E[sel].reshape(len(sel) * b.J, b.K * b.d))
    return torch.cat(out, 0)


def probe_r2(b: Bundle, E, alphas: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)) -> Dict:
    """Ridge probe (fit on train rows, ridge strength chosen per target on val) of the concepts from
    the concatenated slot embeddings; returns R² on train and test per concept group."""
    X = b.xnum_std()
    # Address the two sign-constrained attributes through the metadata rather than by name, and
    # drop any concept with no variance on this dataset (a catalogue dataset has no alternative
    # identity to probe, and may have no time attribute at all).
    t_i, c_i = int(b.meta["time_idx"]), int(b.meta["cost_idx"])
    oh = b.alt_onehot()
    groups = {"time": X[:, :, t_i:t_i + 1], "cost": X[:, :, c_i:c_i + 1], "alt_onehot": oh,
              "Z": b.Z[:, None, :].expand(-1, b.J, -1)}
    groups = {k: g for k, g in groups.items() if float(g.float().std()) > 1e-8}
    if not groups:
        return {"note": "no concept group has variance on this dataset"}
    Y_all = torch.cat(list(groups.values()), -1)                        # (N, J, T)
    sizes = [g.shape[-1] for g in groups.values()]
    idx = {s: b.idx(s) for s in ("train", "val", "test")}
    Xs = {s: _flat_E(E, b, idx[s]).double() for s in idx}
    Ys = {s: Y_all[idx[s]].reshape(-1, Y_all.shape[-1]).double() for s in idx}
    mx, my = Xs["train"].mean(0), Ys["train"].mean(0)
    Xc = {s: Xs[s] - mx for s in Xs}
    Yc = {s: Ys[s] - my for s in Ys}
    G = Xc["train"].T @ Xc["train"]
    XtY = Xc["train"].T @ Yc["train"]
    scale = torch.trace(G) / G.shape[0]
    eye = torch.eye(G.shape[0], dtype=G.dtype)

    def r2(pred, y):
        ss_res = ((y - pred) ** 2).sum(0)
        ss_tot = ((y - y.mean(0)) ** 2).sum(0)
        return torch.where(ss_tot > 1e-12, 1 - ss_res / ss_tot, torch.zeros_like(ss_tot))

    best_val = torch.full((Y_all.shape[-1],), -float("inf"), dtype=torch.float64)
    res = {s: torch.zeros(Y_all.shape[-1], dtype=torch.float64) for s in ("train", "test")}
    acc = {"train": 0.0, "test": 0.0}
    # slice of the alternative-identity block, when this dataset has one
    keys = list(groups)
    if "alt_onehot" in keys:
        k = keys.index("alt_onehot")
        a0 = sum(sizes[:k]); a1 = a0 + sizes[k]
    else:
        a0 = a1 = 0
    for al in alphas:
        Wp = torch.linalg.solve(G + al * scale * eye, XtY)
        pv = r2(Xc["val"] @ Wp, Yc["val"])
        better = pv > best_val
        if not better.any():
            continue
        best_val = torch.where(better, pv, best_val)
        for s in ("train", "test"):
            P = Xc[s] @ Wp
            res[s] = torch.where(better, r2(P, Yc[s]), res[s])
            if a1 > a0 and better[a0:a1].any():
                acc[s] = float((P[:, a0:a1].argmax(1) == Yc[s][:, a0:a1].argmax(1)).double().mean())
    out = {}
    off = 0
    for (name, g), n in zip(groups.items(), sizes):
        out[name] = {s: float(res[s][off:off + n].mean()) for s in ("train", "test")}
        off += n
    if "alt_onehot" in out:
        out["alt_onehot"]["acc_test"] = acc["test"]
        out["alt_onehot"]["acc_train"] = acc["train"]
    return out


# ----------------------------------------------------------------------------- residual member
class ResidualMember(nn.Module):
    """PrefBranch whose utilities are offset by nu . L (fixed structural log-probabilities)."""

    def __init__(self, branch: PrefBranch, offset: torch.Tensor, nu: float):
        super().__init__()
        self.branch = branch
        self.register_buffer("offset", offset.clone())
        self.nu = float(nu)
        self.T = branch.T

    def utilities(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return self.branch.utilities(b, idx) + self.nu * self.offset[idx][:, None, :]

    def probe_logits(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return self.branch.probe_logits(b, idx)

    def semantic_only(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        """The residual read alone (no offset), for diagnostics."""
        return self.branch(b, idx)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        V = self.utilities(b, idx)
        return _log_mean_exp(torch.log_softmax(V, -1), 1)
