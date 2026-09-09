"""Preference-aligned semantic branch + probability-level type mixture (pref_* variants).

Registered models
-----------------
``pref_dual``      structural MNL (pre-fit, frozen beta) + two semantic paths:
                   (i) a residual branch trained log-linearly on the structural residual
                       (utility = lin + g * sem_res, g zero-init and re-fit after training);
                   (ii) an ensemble of 3 preference-aligned branches pre-trained as
                        semantic-only models, mixed at probability level:
                        p = (1 - pi) * softmax(lin + g * sem_res) + pi * mean_m p_sem_m
                        (pi = sigmoid(gamma), gamma = -6 at init, i.e. a ~zero-init gate).
                   The two scalars (g, gamma) are stacked on the validation split (L-BFGS).
``pref_mix``       path (ii) only, 5 members (best on Optima / LPMC, but it cannot use the
                   Swissmetro residual signal that hybrid_base's log-linear branch finds).
``pref_*_sem``     the semantic-only model of each: the ensemble alone (no structural part).
``pref_mnl_cal``   control: MNL + temperature + uniform mixing, same scalar freedom, no semantics.
``pref_mnl_unif``  control: MNL + uniform mixing only.

The preference-aligned branch (``PrefBranch``)
----------------------------------------------
1. Low-rank projection W: d -> r (r = 32) + LayerNorm on every outcome-sentence embedding, with a
   linear probe over the mean-of-K projected sentence trained by an InfoNCE / conditional-logit
   loss within the choice set (chosen vs unchosen), so the projected space is discriminative.
2. Attention over the K sentences: person-agnostic salience (``attn="salience"``, used by the
   final models) or person-conditioned (query from Z, keys from the projected sentences,
   ``attn="person"``); heads are linear r -> M; person weights over the M heads from Z.
3. RUMnet-lite latent types (T learned type vectors FiLM-scale the heads and shift the mixing
   weights; probability = average over types of the softmax).  Inside one branch T = 4 / 8 did not
   beat T = 1; the type mixture that works is across *members* and with the structural MNL as a type.
4. Regularisation: sentence-slot dropout 0.15, projection dropout 0.1, weight decay 1e-3, ~27k params.

What was tried (all under ``TRIED``; runnable with make_builder / make_mix_builder / make_dual_builder)
------------------------------------------------------------------------------------------------------
* pretrain-then-gate (log-linear gate, ``mode="pretrain"``): the branch learns what the MNL knows,
  the gate stays ~0 on Optima/LPMC and the Swissmetro increment of hybrid_base is lost.
* residual training of the branch (``mode="residual"``): reproduces hybrid_base (Swissmetro
  -0.01), nothing on Optima/LPMC (val early-stopping cannot see <0.01-nat increments).
* probability-level mixing of the pre-fit MNL with the pretrained branch: -0.02..-0.04 nats on
  Optima and LPMC; on LPMC the gain is mostly calibration (MNL temperature alone matches it), on
  Optima it survives both calibration controls.  Ensembles of 3-5 members help; T > 1 inside a
  member, person-conditioned attention, MLP heads, MNL temperature and fine-tuning after stacking
  did not.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import evaluate, fit
from experiments.models.baseline import LinearPart, SemanticBranch


def _log_mean_exp(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(x, dim) - math.log(x.shape[dim])


class PrefBranch(nn.Module):
    """Semantic branch on a preference-aligned projection.

    ``utilities(b, idx)`` returns per-type semantic utilities (n, T, J);
    ``forward`` returns the semantic-only log-probabilities (n, J), i.e. the
    log of the type-averaged softmax (for T = 1 these are ordinary logits up to
    a constant, so ``F.cross_entropy`` on them is exact).
    """

    def __init__(self, b: Bundle, *, r: int = 32, M: int = 5, T: int = 1, attn: str = "person",
                 head_hidden: int = 0, w_hidden: int = 32, slot_drop: float = 0.15, person_input: str = "Z",
                 proj_drop: float = 0.1, weights: str = "net", proj: bool = True, probe: bool = True):
        """``weights``: "net" (person weights over the M heads from z_i) or "uniform" (1/M, ablation);
        ``proj=False`` drops the low-rank projection (heads act on the LayerNormed d-dim embedding);
        ``probe=False`` drops the InfoNCE probe (``probe_logits`` returns None, so no auxiliary loss)."""
        super().__init__()
        if not proj:
            r = b.d
        self.r, self.M, self.T, self.attn, self.slot_drop = r, M, T, attn, slot_drop
        self.person_input = person_input
        self.weights_mode, self.has_probe = weights, probe
        pz = b.Z.shape[1] if person_input == "Z" else b.z_d.shape[1]
        self.proj = nn.Linear(b.d, r, bias=False) if proj else nn.Identity()
        self.norm = nn.LayerNorm(r)
        self.proj_drop = nn.Dropout(proj_drop)
        if head_hidden > 0:
            self.heads = nn.Sequential(nn.Linear(r, head_hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(head_hidden, M))
        else:
            self.heads = nn.Linear(r, M)
        if attn == "person":
            self.q = nn.Linear(pz, r)
            self.k = nn.Linear(r, r, bias=False)
        elif attn == "salience":
            self.sal = nn.Linear(r, 1)
        elif attn == "mean":
            pass
        else:
            raise ValueError(attn)
        if weights == "net":
            self.weights = nn.Sequential(nn.Linear(pz, w_hidden), nn.ReLU(), nn.Linear(w_hidden, M))
        elif weights == "uniform":
            self.weights = None
        else:
            raise ValueError(weights)
        # linear probe on the mean projected sentence (InfoNCE target)
        self.probe = nn.Linear(r, 1) if probe else None
        if T > 1:
            self.type_vec = nn.Parameter(torch.randn(T, r) * 0.5)
            self.type_film = nn.Linear(r, M)
            self.type_wbias = nn.Linear(r, M)
            nn.init.zeros_(self.type_film.bias); nn.init.zeros_(self.type_wbias.bias)

    # ---- pieces ----
    def person(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return (b.Z if self.person_input == "Z" else b.z_d)[idx]

    def project(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        H = self.norm(self.proj(b.E[idx]))               # (n, J, K, r)
        return self.proj_drop(H)

    def pool(self, H: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        n, J, K, r = H.shape
        if self.attn == "person":
            q = self.q(z)[:, None, None, :]                       # (n,1,1,r)
            scores = (self.k(H) * q).sum(-1) / math.sqrt(r)       # (n,J,K)
        elif self.attn == "salience":
            scores = self.sal(H).squeeze(-1)
        else:
            scores = torch.zeros(n, J, K, device=H.device)
        if self.training and self.slot_drop > 0:
            keep = (torch.rand(n, J, K, device=H.device) > self.slot_drop)
            keep = keep | (~keep.any(-1, keepdim=True))           # never drop all K
            scores = scores.masked_fill(~keep, -1e4)
        S = torch.softmax(scores, dim=2)
        return (H * S[..., None]).sum(2)                          # (n, J, r)

    def utilities(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        z = self.person(b, idx)
        H = self.project(b, idx)
        Hp = self.pool(H, z)                                      # (n,J,r)
        A = self.heads(Hp)                                        # (n,J,M)
        wl = self.weights(z) if self.weights is not None else torch.zeros(len(z), self.M, device=A.device)  # (n,M)
        if self.T == 1:
            w = torch.softmax(wl, -1)
            return (A * w[:, None, :]).sum(-1)[:, None, :]        # (n,1,J)
        scale = 1.0 + torch.tanh(self.type_film(self.type_vec))   # (T,M)
        wb = self.type_wbias(self.type_vec)                       # (T,M)
        A_t = A[:, None] * scale[None, :, None, :]                # (n,T,J,M)
        w_t = torch.softmax(wl[:, None, :] + wb[None], -1)        # (n,T,M)
        return (A_t * w_t[:, :, None, :]).sum(-1)                 # (n,T,J)

    def probe_logits(self, b: Bundle, idx: torch.Tensor) -> Optional[torch.Tensor]:
        if self.probe is None:
            return None
        H = self.project(b, idx)
        return self.probe(H.mean(2)).squeeze(-1)                  # (n,J)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        V = self.utilities(b, idx)
        return _log_mean_exp(torch.log_softmax(V, -1), 1)         # (n,J) log-probs


class OmleuBranchT1(nn.Module):
    """The original OM-LEU branch (baseline.SemanticBranch) in the (n,T=1,J) interface — protocol ablation."""

    def __init__(self, b: Bundle, **kw):
        super().__init__()
        self.sem = SemanticBranch(b, **kw)
        self.T = 1

    def utilities(self, b, idx):
        return self.sem(b, idx)[:, None, :]

    def probe_logits(self, b, idx):
        return None

    def forward(self, b, idx):
        return torch.log_softmax(self.sem(b, idx), -1)


class PrefHybrid(nn.Module):
    def __init__(self, b: Bundle, branch: nn.Module, gate_init: float = 0.0):
        super().__init__()
        self.lin = LinearPart(b)
        self.sem = branch
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, b, idx):
        V = self.sem.utilities(b, idx)                            # (n,T,J)
        L = self.lin(b, idx)[:, None, :] + self.gate * V
        return _log_mean_exp(torch.log_softmax(L, -1), 1)         # (n,J) log-probs


# ----------------------------------------------------------------------------
# training protocol
# ----------------------------------------------------------------------------

def _aux_loss(branch: nn.Module, lam_probe: float):
    def f(model, b, sel):
        loss = F.cross_entropy(branch(b, sel), b.y[sel])
        pl = branch.probe_logits(b, sel) if lam_probe > 0 else None
        if pl is not None:
            loss = loss + lam_probe * F.cross_entropy(pl, b.y[sel])
        return loss
    return f


def _sem_metrics(branch: nn.Module, b: Bundle) -> Dict:
    m = evaluate(branch, b, "test")
    return {k: m[k] for k in ("top1", "nll", "brier", "ece")}


def make_builder(branch_kw: Optional[dict] = None, *, sem_only: bool = False, branch_cls=PrefBranch,
                 mode: str = "residual", lr_pre: float = 1e-3, lr_joint: float = 3e-4, lr_gate: float = 0.05,
                 wd: float = 1e-3, lam_aux: float = 0.5, lam_probe: float = 0.5, patience_pre: int = 8,
                 patience_joint: int = 10, max_epochs: int = 80, joint: str = "all"):
    """``mode="pretrain"``: stage 2 fits the branch alone to the choices, stage 3 gate + branch on the hybrid.
    ``mode="residual"``: stage 2 fits the branch on CE(lin_frozen + sem) (gate held at 1), stage 3 re-fits the
    zero-initialised gate (fast lr) with the branch either frozen (``joint="gate"``) or fine-tuned (``joint="all"``)."""
    branch_kw = dict(branch_kw or {})

    def build(b: Bundle, seed: int):
        torch.manual_seed(seed)
        branch = branch_cls(b, **branch_kw)
        aux = _aux_loss(branch, lam_probe)
        has_probe = lam_probe > 0 and branch.probe_logits(b, b.idx("train")[:2]) is not None
        probe_only = (lambda model, b, sel: lam_probe * F.cross_entropy(branch.probe_logits(b, sel), b.y[sel])) \
            if has_probe else None
        if sem_only:
            def run(model, b, seed):
                fr = fit(model, b, params=list(model.parameters()), lr=lr_pre, max_epochs=max_epochs,
                         patience=patience_pre, batch_size=256, weight_decay=wd, seed=seed, extra_loss=probe_only)
                return {"fit": fr.__dict__, "extra": {"n_sem_params": sum(p.numel() for p in model.parameters())}}
            return branch, run
        model = PrefHybrid(b, branch)

        def run(model, b, seed):
            ce = model.lin.prefit(b)
            bp = list(branch.parameters())
            if mode == "pretrain":
                fr_pre = fit(branch, b, params=bp, lr=lr_pre, max_epochs=max_epochs, patience=patience_pre,
                             batch_size=256, weight_decay=wd, seed=seed, extra_loss=probe_only)
                sem_only_m = _sem_metrics(branch, b)
                joint_params = [{"params": [model.gate], "lr": lr_gate, "weight_decay": 0.0},
                                {"params": bp, "lr": lr_joint}]
                fr = fit(model, b, params=joint_params, lr=lr_joint, max_epochs=max_epochs, patience=patience_joint,
                         batch_size=256, weight_decay=wd, seed=seed + 1,
                         extra_loss=(lambda m_, b_, s_: lam_aux * aux(m_, b_, s_)) if lam_aux > 0 else None)
            elif mode == "residual":
                model.gate.data.fill_(1.0)
                fr_pre = fit(model, b, params=bp, lr=lr_pre, max_epochs=max_epochs, patience=patience_pre,
                             batch_size=256, weight_decay=wd, seed=seed, extra_loss=probe_only)
                sem_only_m = _sem_metrics(branch, b)     # the residual branch read alone (not a fitted sem-only model)
                model.gate.data.zero_()
                joint_params = [{"params": [model.gate], "lr": lr_gate, "weight_decay": 0.0}]
                if joint == "all":
                    joint_params.append({"params": bp, "lr": lr_joint})
                fr = fit(model, b, params=joint_params, lr=lr_gate, max_epochs=max_epochs, patience=patience_joint,
                         batch_size=256, weight_decay=wd, seed=seed + 1, extra_loss=probe_only if joint == "all" else None)
            else:
                raise ValueError(mode)
            return {"fit": {**fr.__dict__, "stage1_train_ce": ce, "sem_pre_best_val": fr_pre.best_val_nll,
                            "sem_pre_best_epoch": fr_pre.best_epoch, "mode": mode},
                    "extra": {"gate": float(model.gate), "sem_only_test": sem_only_m,
                              "n_sem_params": sum(p.numel() for p in branch.parameters())}}
        return model, run
    return build


# ----------------------------------------------------------------------------
# probability-level mixture: structural MNL as one latent type, semantic members as the others
# ----------------------------------------------------------------------------

class MixHybrid(nn.Module):
    """p(j) = (1 - pi) * softmax(a * lin)(j) + pi * mean_members p_sem(j).

    ``pi = sigmoid(gamma)`` is the gate (gamma = -6 at init, pi = 0.0025, i.e. the model starts as the
    structural MNL); ``a = exp(log_a)`` is an optional temperature on the pre-fit MNL (beta fixed, so
    the marginal rates of substitution are untouched).  With ``members`` empty the model is the
    calibration control (MNL + temperature + uniform mixing), no semantics.
    """

    def __init__(self, b: Bundle, members: list, *, temp: bool = False, gamma_init: float = -6.0,
                 res: Optional[nn.Module] = None):
        super().__init__()
        self.lin = LinearPart(b)
        self.members = nn.ModuleList(members)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.log_a = nn.Parameter(torch.zeros(()), requires_grad=temp)
        self.temp = temp
        self.J = b.J
        self.res = res                                   # optional residual-trained semantic branch (log-linear)
        self.g = nn.Parameter(torch.tensor(0.0))         # its zero-init gate

    def structural(self, b, idx):
        u = self.lin(b, idx)
        if self.res is not None:
            u = u + self.g * self.res.utilities(b, idx)[:, 0, :]
        return u

    @property
    def gate(self):
        return torch.sigmoid(self.gamma)

    def sem_logp(self, b, idx):
        if len(self.members) == 0:
            return torch.full((len(idx), self.J), -math.log(self.J), device=b.y.device)
        return _log_mean_exp(torch.stack([m(b, idx) for m in self.members], 1), 1)

    def forward(self, b, idx):
        a = self.log_a.exp() if self.temp else 1.0
        Lp = torch.log_softmax(a * self.structural(b, idx), -1)
        Sp = self.sem_logp(b, idx)
        pi = self.gate
        return torch.logaddexp(torch.log1p(-pi) + Lp, torch.log(pi) + Sp)


class SemEnsemble(nn.Module):
    """Semantic-only counterpart of MixHybrid: type-averaged probability of the members."""

    def __init__(self, members: list):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, b, idx):
        return _log_mean_exp(torch.stack([m(b, idx) for m in self.members], 1), 1)


def make_mix_builder(branch_kw: Optional[dict] = None, *, members: int = 1, sem_only: bool = False, temp: bool = False,
                     stack: str = "val", lr_pre: float = 1e-3, wd: float = 1e-3, lam_probe: float = 0.5,
                     patience_pre: int = 8, max_epochs: int = 80, finetune: bool = False, lr_ft: float = 1e-4):
    """``stack="val"``: the scalars (gamma, log_a) are fit on the validation split by L-BFGS (stacked
    generalisation / temperature scaling); ``stack="train"``: fit on train by Adam with early stopping on val.
    ``finetune``: after stacking, fine-tune the members on the mixture objective (train, early stop on val)."""
    branch_kw = dict(branch_kw or {})

    def pretrain(member, b, seed):
        probe = (lambda m_, b_, s_: lam_probe * F.cross_entropy(member.probe_logits(b_, s_), b_.y[s_])) if lam_probe > 0 else None
        return fit(member, b, params=list(member.parameters()), lr=lr_pre, max_epochs=max_epochs, patience=patience_pre,
                   batch_size=256, weight_decay=wd, seed=seed, extra_loss=probe)

    def build(b: Bundle, seed: int):
        torch.manual_seed(seed)
        mems = [PrefBranch(b, **branch_kw) for _ in range(members)]
        if sem_only:
            model = SemEnsemble(mems)

            def run(model, b, seed):
                frs = [pretrain(m, b, seed * 100 + i) for i, m in enumerate(mems)]
                return {"fit": {**frs[0].__dict__, "member_best_val": [f.best_val_nll for f in frs]},
                        "extra": {"n_sem_params": sum(p.numel() for p in model.parameters()), "members": members}}
            return model, run
        model = MixHybrid(b, mems, temp=temp)

        def run(model, b, seed):
            ce = model.lin.prefit(b)
            frs = [pretrain(m, b, seed * 100 + i) for i, m in enumerate(mems)]
            sem_m = _sem_metrics(SemEnsemble(mems), b) if members else None
            scalars = [model.gamma] + ([model.log_a] if temp else [])
            if stack == "val":
                va = b.idx("val")
                model.eval()
                from experiments.harness.train import lbfgs_prefit
                val_nll = lbfgs_prefit(lambda: F.cross_entropy(model(b, va), b.y[va]), scalars, max_iter=200)
                fr = {"best_epoch": 0, "best_val_nll": val_nll, "epochs": 0, "seconds": 0.0, "val_curve": []}
            else:
                fr_ = fit(model, b, params=[{"params": scalars, "lr": 0.05, "weight_decay": 0.0}], lr=0.05,
                          max_epochs=max_epochs, patience=10, batch_size=256, seed=seed + 1)
                fr = fr_.__dict__
            if finetune and members:
                fr_ft = fit(model, b, params=[p for m in mems for p in m.parameters()], lr=lr_ft, max_epochs=max_epochs,
                            patience=8, batch_size=256, weight_decay=wd, seed=seed + 2)
                fr = {**fr, "finetune_best_val": fr_ft.best_val_nll, "finetune_best_epoch": fr_ft.best_epoch}
            return {"fit": {**fr, "stage1_train_ce": ce, "member_best_val": [f.best_val_nll for f in frs]},
                    "extra": {"gate": float(model.gate), "temp_a": float(model.log_a.exp()), "sem_only_test": sem_m,
                              "n_sem_params": sum(p.numel() for m in mems for p in m.parameters()), "members": members}}
        return model, run
    return build


def make_dual_builder(branch_kw: Optional[dict] = None, *, members: int = 3, res_cls=OmleuBranchT1, res_kw: Optional[dict] = None,
                      stack: str = "val", lr_pre: float = 1e-3, wd: float = 1e-3, lam_probe: float = 0.5,
                      patience_pre: int = 8, max_epochs: int = 80, lr_res: float = 1e-3, wd_res: float = 1e-4,
                      ridge: float = 1e-3):
    """Two semantic paths: (i) a residual branch trained log-linearly on the structural residual (g held at 1
    during its fit, then re-fit from 0), (ii) a pretrained ensemble mixed at probability level (pi from ~0).
    The two scalars (g, gamma) are stacked on val (L-BFGS) or fit on train (Adam, early stop on val)."""
    branch_kw = dict(branch_kw or {}); res_kw = dict(res_kw or {})

    def pretrain(member, b, seed):
        probe = (lambda m_, b_, s_: lam_probe * F.cross_entropy(member.probe_logits(b_, s_), b_.y[s_])) if lam_probe > 0 else None
        return fit(member, b, params=list(member.parameters()), lr=lr_pre, max_epochs=max_epochs, patience=patience_pre,
                   batch_size=256, weight_decay=wd, seed=seed, extra_loss=probe)

    def build(b: Bundle, seed: int):
        torch.manual_seed(seed)
        mems = [PrefBranch(b, **branch_kw) for _ in range(members)]
        res = res_cls(b, **res_kw)
        model = MixHybrid(b, mems, res=res)

        def run(model, b, seed):
            ce = model.lin.prefit(b)
            # residual path: lin + 1 * res, only res trained
            model.g.data.fill_(1.0)
            fr_res = fit(model, b, params=list(res.parameters()), lr=lr_res, max_epochs=max_epochs, patience=patience_pre,
                         batch_size=256, weight_decay=wd_res, seed=seed + 3)
            res_only_m = _sem_metrics(res, b)
            model.g.data.zero_()
            frs = [pretrain(m, b, seed * 100 + i) for i, m in enumerate(mems)]
            sem_m = _sem_metrics(SemEnsemble(mems), b)
            # an untrained residual branch (no validation improvement) is noise: keep its gate at 0
            res_used = fr_res.best_epoch > 0
            scalars = ([model.g] if res_used else []) + [model.gamma]
            if stack == "val":
                va = b.idx("val"); model.eval()
                from experiments.harness.train import lbfgs_prefit
                val_nll = lbfgs_prefit(lambda: F.cross_entropy(model(b, va), b.y[va]) + ridge * model.g ** 2,
                                       scalars, max_iter=200)
                fr = {"best_epoch": 0, "best_val_nll": val_nll, "epochs": 0, "seconds": 0.0, "val_curve": []}
            else:
                fr = fit(model, b, params=[{"params": scalars, "lr": 0.05, "weight_decay": 0.0}], lr=0.05,
                         max_epochs=max_epochs, patience=10, batch_size=256, seed=seed + 1).__dict__
            return {"fit": {**fr, "stage1_train_ce": ce, "res_best_val": fr_res.best_val_nll, "res_best_epoch": fr_res.best_epoch,
                            "member_best_val": [f.best_val_nll for f in frs]},
                    "extra": {"gate": float(model.gate), "res_gate": float(model.g), "res_used": res_used, "sem_only_test": sem_m,
                              "res_only_test": res_only_m, "members": members,
                              "n_sem_params": sum(p.numel() for m in list(mems) + [res] for p in m.parameters())}}
        return model, run
    return build


# ----------------------------------------------------------------------------
# registered variants (see experiments/models/__init__.py) and tried configurations
# ----------------------------------------------------------------------------
V0 = dict()                                                                   # original OM-LEU branch (baseline.SemanticBranch)
V1 = dict(r=32, M=5, T=1, attn="salience", head_hidden=0, slot_drop=0.15)     # projection + probe + linear heads
V2 = dict(r=32, M=5, T=1, attn="person", head_hidden=0, slot_drop=0.15)       # + person-conditioned attention
V3 = dict(r=32, M=5, T=4, attn="person", head_hidden=0, slot_drop=0.15)       # + latent-type mixture T=4
V3_T8 = dict(r=32, M=5, T=8, attn="person", head_hidden=0, slot_drop=0.15)
V4 = dict(r=64, M=8, T=4, attn="person", head_hidden=32, slot_drop=0.15)      # larger, MLP heads

# final
build_pref_dual = make_dual_builder(V1, members=3)
build_pref_dual_sem = make_mix_builder(V1, members=3, sem_only=True)
build_pref_mix = make_mix_builder(V1, members=5)
build_pref_mix_sem = make_mix_builder(V1, members=5, sem_only=True)
build_pref_mnl_cal = make_mix_builder(members=0, temp=True)
build_pref_mnl_unif = make_mix_builder(members=0, temp=False)

# tried and not kept (name -> builder); numbers in the session report
TRIED = {
    "pref_v0": make_builder(V0, branch_cls=OmleuBranchT1, lam_probe=0.0, mode="pretrain"),
    "pref_v1": make_builder(V1, mode="pretrain"), "pref_v2": make_builder(V2, mode="pretrain"),
    "pref_v3": make_builder(V3, mode="pretrain"), "pref_v3_t8": make_builder(V3_T8, mode="pretrain"),
    "pref_v4": make_builder(V4, mode="pretrain"),
    "pref_r0": make_builder(V0, branch_cls=OmleuBranchT1, lam_probe=0.0, mode="residual"),
    "pref_r1": make_builder(V1, mode="residual"), "pref_r2": make_builder(V2, mode="residual"),
    "pref_r3": make_builder(V3, mode="residual"), "pref_r3_t8": make_builder(V3_T8, mode="residual"),
    "pref_r4": make_builder(V4, mode="residual"), "pref_r2_gate": make_builder(V2, mode="residual", joint="gate"),
    "pref_r2_noprobe": make_builder(V2, mode="residual", lam_probe=0.0),
    "pref_mix1": make_mix_builder(V1, members=1), "pref_mix2": make_mix_builder(V2, members=1),
    "pref_mix2T": make_mix_builder(V2, members=1, temp=True), "pref_mix2e3": make_mix_builder(V2, members=3),
    "pref_mix2e3T": make_mix_builder(V2, members=3, temp=True), "pref_mix1e3": make_mix_builder(V1, members=3),
    "pref_mix3e3": make_mix_builder(V3, members=3), "pref_mix2e3_train": make_mix_builder(V2, members=3, stack="train"),
    "pref_mix2e3_ft": make_mix_builder(V2, members=3, finetune=True),
    "pref_mix1e3_train": make_mix_builder(V1, members=3, stack="train"),
    "pref_dual_train": make_dual_builder(V1, members=3, stack="train"),
    "pref_dual_r2": make_dual_builder(V1, members=3, res_cls=PrefBranch, res_kw=V2, wd_res=1e-3),
}
