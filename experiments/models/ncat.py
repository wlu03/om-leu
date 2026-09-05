"""Numeric-aware concept heads (NCAT) — OM-LEU variants.

Semantic branch on top of the pre-fit linear MNL (``LinearPart``), behind a
zero-initialised gate, with three changes:

* **magnitude channel**: every sentence embedding is concatenated with the
  standardised numeric attributes of its alternative (``bundle.xnum_std()``),
  the alternative one-hot, and optionally Fourier features of the numbers
  (FoNE style), so a head can compute "how much time × how I feel about it";
* **block-diagonal concept heads**: the K = 5 sentences are axis-anchored
  (financial, time, comfort, convenience, reliability), so head m reads slot m
  only and returns one concept score per axis (a concept bottleneck); an
  optional shared head reads the mean sentence; ``block=False`` restores the
  free heads + salience of ``hybrid_base`` (with the same magnitude channel)
  as a control;
* **person weights over concept scores** from z_d (or Z), optionally with a
  learned interaction between each concept score and a linear function of the
  numeric attributes (zero-initialised).

Capacity is kept small (linear or 32-hidden heads, optional PCA projection of
the 768-d embedding fit on train, dropout) and the semantic part is trained
with early stopping on validation NLL.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import fit, lbfgs_prefit
from experiments.models.baseline import LinearPart, mlp


@torch.no_grad()
def _all_logits(model: nn.Module, b: Bundle) -> torch.Tensor:
    idx = torch.arange(b.N)
    return torch.cat([model(b, idx[s:s + 4096]) for s in range(0, b.N, 4096)], 0)


def oof_linear_logits(b: Bundle, seed: int, folds: int = 5, l2: float = 1e-4) -> torch.Tensor:
    """Structural logits with train rows predicted out-of-fold (folds grouped by person); val/test
    rows from the full-train fit.  The in-sample pre-fit overfits train (Optima: train CE 0.37 vs
    val 0.49), leaving no residual for the semantic branch to learn; cross-fitting restores it."""
    full = LinearPart(b); full.prefit(b, l2=l2)
    out = _all_logits(full, b).clone()
    tr = b.idx("train")
    persons = b.person[tr]
    uniq = persons.unique()
    g = torch.Generator().manual_seed(seed)
    fold_of = torch.zeros(int(uniq.max()) + 1, dtype=torch.long)
    fold_of[uniq[torch.randperm(len(uniq), generator=g)]] = torch.arange(len(uniq)) % folds
    fold = fold_of[persons]
    for f in range(folds):
        hold, fit_idx = tr[fold == f], tr[fold != f]
        m = LinearPart(b)

        def loss(m=m, fit_idx=fit_idx):
            return F.cross_entropy(m(b, fit_idx), b.y[fit_idx]) + l2 * (m.beta ** 2).sum()
        lbfgs_prefit(loss, [m.beta])
        with torch.no_grad():
            out[hold] = m(b, hold)
    return out


class BlockHeads(nn.Module):
    """M independent heads applied to M inputs (batched): x (..., M, n_in) -> (..., M)."""

    def __init__(self, M: int, n_in: int, hidden: int = 32, dropout: float = 0.2, linear: bool = False,
                 zero_out: bool = False):
        super().__init__()
        self.linear = linear
        bound = 1.0 / math.sqrt(n_in)
        if linear:
            self.W = nn.Parameter(torch.empty(M, n_in).uniform_(-bound, bound))
            self.b = nn.Parameter(torch.zeros(M))
        else:
            self.W1 = nn.Parameter(torch.empty(M, n_in, hidden).uniform_(-bound, bound))
            self.b1 = nn.Parameter(torch.zeros(M, hidden))
            b2 = 1.0 / math.sqrt(hidden)
            self.W2 = nn.Parameter(torch.empty(M, hidden).uniform_(-b2, b2))
            self.b2 = nn.Parameter(torch.zeros(M))
            self.drop = nn.Dropout(dropout)
        if zero_out:                                   # branch output is exactly 0 at init
            with torch.no_grad():
                (self.W if linear else self.W2).zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear:
            return (x * self.W).sum(-1) + self.b
        h = torch.einsum("...mi,mih->...mh", x, self.W1) + self.b1
        h = self.drop(F.relu(h))
        return (h * self.W2).sum(-1) + self.b2


class ConceptBranch(nn.Module):
    def __init__(self, b: Bundle, *, block: bool = True, head: str = "mlp", hidden: int = 32, dropout: float = 0.2,
                 pca: Optional[int] = 64, num_channel: str = "raw", fourier_freqs=(0.5, 1.0, 2.0),
                 shared_head: bool = False, interact: bool = False, weights_from: str = "z_d", w_hidden: int = 32,
                 use_emb: bool = True, in_dropout: float = 0.0, s_hidden: int = 32, zero_readout: bool = False,
                 weight_kind: str = "softmax", z_in_heads: bool = False):
        super().__init__()
        self.weight_kind, self.z_in_heads = weight_kind, z_in_heads
        self.block, self.use_emb, self.interact, self.shared_head = block, use_emb, interact, shared_head
        self.num_channel, self.freqs = num_channel, tuple(fourier_freqs)
        K, A, Fn = b.K, b.n_alts, b.F
        # ---- magnitude channel (buffer, standardised on train) ----
        xs = b.xnum_std()                                                 # (N, J, F)
        oh = b.alt_onehot()                                               # (N, J, A)
        parts = [oh]
        if num_channel in ("raw", "fourier"):
            parts.append(xs)
        if num_channel == "fourier":
            for f in self.freqs:
                parts += [torch.sin(2 * math.pi * f * xs), torch.cos(2 * math.pi * f * xs)]
        self.register_buffer("C", torch.cat(parts, -1) if num_channel != "none" else oh)   # (N, J, c)
        self.register_buffer("xs", xs)
        c_dim = self.C.shape[-1]
        # ---- embedding projection (PCA on train, fixed) ----
        e_dim = 0
        if use_emb and pca:
            tr = b.idx("train")
            flat = b.E[tr].reshape(-1, b.d)
            mu = flat.mean(0)
            cov = (flat - mu).T @ (flat - mu) / flat.shape[0]
            evals, evecs = torch.linalg.eigh(cov)
            comp = evecs[:, -pca:].flip(-1)                                # (d, pca)
            scale = evals[-pca:].flip(0).clamp_min(1e-8).sqrt()
            self.register_buffer("pca_mu", mu)
            self.register_buffer("pca_W", comp / scale)                     # whitened scores
            e_dim = pca
        else:
            self.pca_W = None
            e_dim = b.d if use_emb else 0
        self.e_dim = e_dim
        self.in_drop = nn.Dropout(in_dropout) if in_dropout > 0 else nn.Identity()
        n_in = e_dim + c_dim + (b.z_d.shape[1] if z_in_heads else 0)
        M = K if block else 5
        self.M = M
        if block:
            self.heads = BlockHeads(M, n_in, hidden, dropout, linear=(head == "linear"), zero_out=zero_readout)
        else:
            self.heads = (nn.Linear(n_in, M) if head == "linear" else mlp(n_in, hidden, M, dropout))
            self.salience = mlp(n_in, s_hidden, 1, dropout)
            if zero_readout:
                nn.init.zeros_((self.heads if head == "linear" else self.heads[-1]).weight)
        n_scores = M + (1 if shared_head else 0)
        if shared_head:
            self.shared = (nn.Linear(n_in, 1) if head == "linear" else mlp(n_in, hidden, 1, dropout))
            if zero_readout:
                nn.init.zeros_((self.shared if head == "linear" else self.shared[-1]).weight)
        zin = {"z_d": b.z_d, "Z": b.Z}[weights_from]
        self.weights_from = weights_from
        self.weights = mlp(zin.shape[1], w_hidden, n_scores, dropout=0.0)
        if weight_kind == "signed":                    # w = 1 + f(z), f zero-initialised
            nn.init.zeros_(self.weights[-1].weight); nn.init.zeros_(self.weights[-1].bias)
        if interact:
            self.V = nn.Parameter(torch.zeros(n_scores, Fn))               # score_m × (V_m · x_std)
        self.n_scores = n_scores

    def embed(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        E = b.E[idx]                                                      # (n, J, K, d)
        if self.pca_W is not None:
            E = (E - self.pca_mu) @ self.pca_W
        return E

    def scores(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        """Concept scores (n, J, n_scores)."""
        C = self.C[idx]                                                   # (n, J, c)
        n, J, _ = C.shape
        if self.z_in_heads:
            C = torch.cat([C, b.z_d[idx][:, None, :].expand(-1, J, -1)], -1)
        if self.use_emb:
            E = self.in_drop(self.embed(b, idx))                          # (n, J, K, e)
            X = torch.cat([E, C[:, :, None, :].expand(-1, -1, E.shape[2], -1)], -1)   # (n, J, K, e+c)
        else:
            X = C[:, :, None, :].expand(-1, -1, b.K, -1)
        if self.block:
            S = self.heads(X)                                             # (n, J, M)
        else:
            Aq = self.heads(X)                                            # (n, J, K, M)
            sal = torch.softmax(self.salience(X).squeeze(-1), dim=2)      # (n, J, K)
            S = (Aq * sal[..., None]).sum(2)
        if self.shared_head:
            S = torch.cat([S, self.shared(X.mean(2))], -1)
        return S

    def person_weights(self, zin: torch.Tensor) -> torch.Tensor:
        if self.weight_kind == "signed":
            return 1.0 + self.weights(zin)
        return torch.softmax(self.weights(zin), dim=-1)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        S = self.scores(b, idx)                                           # (n, J, m)
        zin = {"z_d": b.z_d, "Z": b.Z}[self.weights_from][idx]
        w = self.person_weights(zin)                                      # (n, m)
        if self.interact:
            g = 1.0 + self.xs[idx] @ self.V.T                             # (n, J, m)
            S = S * g
        return (S * w[:, None, :]).sum(-1)


class NCAT(nn.Module):
    """logits = β·X_mnl + (1/R) Σ_r gate_r · sem_r.  ``init="gate0"``: gate starts at 0 (hybrid_base);
    ``init="zero_readout"``: gate starts at 1 and the branch's output layer at 0 — the same function
    (exactly the MNL at step 0) but the readout receives a non-zero gradient, which the
    multiplicative zero gate does not (its gradient is gate × ... = 0, and Adam moves the gate by
    ~lr per step, so on Optima it is still 0.03 after 20 epochs).  ``n_restarts`` R > 1 trains R
    independently initialised branches (each with early stopping) and averages their utilities."""

    def __init__(self, b: Bundle, *, init: str = "gate0", n_restarts: int = 1, offset: str = "insample",
                 oof_folds: int = 5, seed: int = 0, **kw):
        super().__init__()
        self.lin = LinearPart(b)
        self.offset = offset
        if offset == "oof":                          # training-time structural offset, cross-fitted
            self.register_buffer("lin_oof", oof_linear_logits(b, seed, oof_folds))
        zr = init == "zero_readout"
        self.sems = nn.ModuleList([ConceptBranch(b, zero_readout=zr, **kw) for _ in range(n_restarts)])
        self.gates = nn.Parameter(torch.full((n_restarts,), 1.0 if zr else 0.0))
        self.active: Optional[int] = None            # train one branch at a time

    @property
    def sem(self):
        return self.sems[0]

    @property
    def gate(self):
        return self.gates[0]

    def sem_utility(self, b, idx):
        if self.active is not None:
            return self.gates[self.active] * self.sems[self.active](b, idx)
        return sum(g * s(b, idx) for g, s in zip(self.gates, self.sems)) / len(self.sems)

    def forward(self, b, idx):
        if self.training and self.offset == "oof":
            base = self.lin_oof[idx]
        else:
            base = self.lin(b, idx)
        return base + self.sem_utility(b, idx)


def _make(train_kw: dict, gate_lr_mult: float = 1.0, **kw):
    def build(b: Bundle, seed: int):
        model = NCAT(b, seed=seed, **kw)

        def run(model, b, seed):
            ce = model.lin.prefit(b)
            tk = dict(train_kw)
            fits = []
            for r in range(len(model.sems)):
                model.active = r
                gate_mask = torch.zeros_like(model.gates); gate_mask[r] = 1.0
                hook = model.gates.register_hook(lambda g, m=gate_mask: g * m)
                pg = [{"params": list(model.sems[r].parameters())},
                      {"params": [model.gates], "lr": tk["lr"] * gate_lr_mult, "weight_decay": 0.0}]
                fits.append(fit(model, b, params=pg, seed=seed * 100 + r, **tk))
                hook.remove()
            model.active = None
            fr = fits[0]
            extra = {"gate": float(model.gates[0]), "gates": model.gates.tolist(),
                     "config": {**kw, **train_kw, "gate_lr_mult": gate_lr_mult},
                     "n_sem_params": int(sum(p.numel() for p in model.sems.parameters())),
                     "restart_best_val": [f.best_val_nll for f in fits], "restart_best_epoch": [f.best_epoch for f in fits]}
            with torch.no_grad():
                te = b.idx("test")
                sem = model.sems[0]
                zin = {"z_d": b.z_d, "Z": b.Z}[sem.weights_from][te]
                w = sem.person_weights(zin)
                extra["mean_weights"] = w.mean(0).tolist()
                extra["weight_std"] = w.std(0).tolist()
                U = torch.cat([model.sem_utility(b, te[s:s + 2048]) for s in range(0, len(te), 2048)], 0)
                U = U - U.mean(1, keepdim=True)
                extra["sem_util_std"] = float(U.std())                      # gated utility spread on test
                L = torch.cat([model.lin(b, te[s:s + 2048]) for s in range(0, len(te), 2048)], 0)
                extra["lin_util_std"] = float((L - L.mean(1, keepdim=True)).std())
                S = torch.cat([sem.scores(b, te[s:s + 2048]) for s in range(0, len(te), 2048)], 0)
                Sf = S.reshape(-1, S.shape[-1])
                extra["score_std"] = Sf.std(0).tolist()
                if S.shape[-1] > 1 and float(Sf.std()) > 0:
                    extra["score_corr"] = torch.corrcoef(Sf.T).nan_to_num().tolist()
                if sem.interact:
                    extra["V"] = sem.V.tolist()
            return {"fit": {**fr.__dict__, "stage1_train_ce": ce, "epochs_total": sum(f.epochs for f in fits),
                            "seconds": sum(f.seconds for f in fits)}, "extra": extra}
        return model, run
    return build


TRAIN = dict(lr=5e-4, max_epochs=80, patience=12, batch_size=256, weight_decay=1e-3)
ZR = dict(init="zero_readout", pca=64)

# ---- registered (best two) ----
# v2: block-diagonal MLP-32 concept heads on [PCA-64 sentence ; alt one-hot ; standardised numbers ; Fourier(numbers)],
#     concept × numeric interaction, shared head, softmax person weights from z_d, cross-fitted structural offset
build_ncat_v2 = _make(TRAIN, **ZR, block=True, head="mlp", hidden=32, dropout=0.2, num_channel="fourier",
                      interact=True, shared_head=True, offset="oof")
# v3: same with linear concept heads (lowest capacity)
build_ncat_v3 = _make(TRAIN, **ZR, block=True, head="linear", num_channel="fourier", interact=True, shared_head=True,
                      offset="oof")

# ---- also tried (results under experiments/results/<name>/; not registered) ----
# v1: block MLP-32 heads, raw numeric channel only, in-sample offset
build_ncat_v1 = _make(TRAIN, **ZR, block=True, head="mlp", hidden=32, dropout=0.2, num_channel="raw")
# v2e: v2 (in-sample offset) with 5 independently trained branches averaged (bagging over restarts)
build_ncat_v2e = _make(TRAIN, **ZR, block=True, head="mlp", hidden=32, dropout=0.2, num_channel="fourier",
                       interact=True, shared_head=True, n_restarts=5)
# free-head control: hybrid_base heads + salience over K, same magnitude channel
build_ncat_free = _make(TRAIN, **ZR, block=False, head="mlp", hidden=32, dropout=0.2, num_channel="raw")
# numeric-only control: no embeddings at all (is the gain just a nonlinear numeric residual?)
build_ncat_numonly = _make(TRAIN, init="zero_readout", pca=None, block=True, head="mlp", hidden=32, dropout=0.2,
                           num_channel="raw", use_emb=False)
# hybrid_base's multiplicative zero gate with the v2 branch (shows the dead gate: best epoch 0 on Optima/LPMC)
build_ncat_gate0 = _make(TRAIN, init="gate0", pca=64, block=True, head="mlp", hidden=32, dropout=0.2,
                         num_channel="fourier", interact=True, shared_head=True)
