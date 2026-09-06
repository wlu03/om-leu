"""OM-LEU 2: heterogeneous structural utility + RUM-shaped boosted residual + semantic mixture.

The combined successor of ``hybrid_base``, assembled from the parts that survived the four
idea branches (hetero_v1, boost_v2, pref_mix, ncat_v3).  For event i (person p, covariates
z_i) and alternative j with attributes x_ij, history h_ij and outcome sentences E_ij:

Stage 1  structural utility (hetero_v1; ``experiments.models.hetero``)

    V_ij^r = ASC_j + beta . X_ij - softplus(r_t + d_t(z_i)) t_ij - softplus(r_c + d_c(z_i)) c_ij
             + g_j(z_i) + u_pj                                    r = 1..R restarts (R = 5)
    L_ij   = log (1/R) sum_r softmax_j(V_i^r)                     ensemble log-probability

    TasteNet time/cost coefficients (< 0), functional intercepts g_j(z), person random
    effects u_pj (L2-shrunk, selected on val; switched off on cold-start splits).

Stage 1c (``ncat=True``) log-linear concept residual (ncat_v3; ``experiments.models.ncat``)

    L_ij <- L_ij + g . C_ij,   C_ij = sum_m w_m(z_i) [1 + V_m . x_ij] s_m(E_ijm, x_ij)

    block-diagonal linear concept heads s_m on [PCA-64 sentence m ; alt one-hot ; standardised
    numbers ; Fourier(numbers)], zero-initialised readout, trained on the cross-fitted structural
    residual with early stopping on val.

Stage 2  boosted residual (boost_v2; ``experiments.models.boost`` / ``_lgb_child.py``)

    U_ij = tau . L_ij + f_j(x_ij, z_i, h_ij)

    one LightGBM booster f_j per alternative on that alternative's own feature block, trained
    jointly under the softmax cross-entropy with tau . L as ``init_score``; monotone -1 on time
    and cost; early stopping on val; tau selected on val from ``tau_grid``.  On train rows the
    offset L is cross-fitted (``oof_folds`` refits of stage 1 on fold complements) so the trees
    see an honest residual rather than the in-sample fit; val/test rows use the full-train fit.
    tau = 1 is admissible here (with the honest offset the trees have a residual to learn even
    without shrinkage) and, with the early-stopped round at 0, reduces exactly to stage 1.

Stage 3  semantic type mixture (pref_mix; ``experiments.models.pref``)

    p_i(j) = (1 - pi) softmax_j(a U_i) + pi (1/M) sum_m p_m^sem(j | E_i, z_i)

    M = 5 ``PrefBranch`` members pretrained as semantic-only models (InfoNCE probe on the
    preference-aligned projection); the two scalars pi = sigmoid(gamma) and a = exp(log_a)
    (temperature on the structural part) are stacked on the validation split by L-BFGS.
    With ``members=0`` and ``temp=True`` the model is the no-semantics calibration control.

Registered variants (``experiments/models/__init__.py``): combo_struct (stages 1-2),
combo_struct_cal (+ temperature, control), combo_full (+ stage 3), combo_full_ncat (+ stage 1c),
combo_struct_ncat and combo_struct_insample (ablations: stage 1c without semantics; in-sample
instead of cross-fitted offset).  Stage 1 and the member pretraining are memoised per
(dataset, seed) inside a process, so running several variants in one ``run.py`` invocation
shares them; every stage re-seeds, so the memoised result equals a fresh computation.
"""
from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import evaluate, fit, lbfgs_prefit, nll_on
from experiments.models.boost import DEFAULT as BOOST_DEFAULT, RUM, _canon, all_logits, build_blocks, run_child
from experiments.models.hetero import build_hetero_v1, make_builder as make_hetero_builder
from experiments.models.ncat import ConceptBranch
from experiments.models.pref import PrefBranch, V1 as PREF_V1, _log_mean_exp

HETERO_V1 = dict(taste="generic", intercepts=True, person=True)        # hetero_v1's structural form
NCAT_V3 = dict(block=True, head="linear", num_channel="fourier", interact=True, shared_head=True, pca=64,
               zero_readout=True)
NCAT_TRAIN = dict(lr=5e-4, max_epochs=80, patience=12, batch_size=256, weight_decay=1e-3)
BOOST_CFG = {**BOOST_DEFAULT, **RUM, "sem": None}                        # boost_v2's booster
PREF_TRAIN = dict(lr=1e-3, max_epochs=80, patience=8, batch_size=256, weight_decay=1e-3)
PREF_LAM_PROBE = 0.5


@dataclass(frozen=True)
class Config:
    oof_folds: int = 5                  # 0: in-sample structural offset for the booster
    tau_grid: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    ncat: bool = False                  # stage 1c concept residual
    members: int = 0                    # stage 3 semantic members (0: none)
    temp: bool = False                  # temperature a on the structural part, stacked on val
    # --- ablation switches (hashable so they can key the stage caches) ---
    struct: Tuple[Tuple[str, object], ...] = ()   # hetero overrides, e.g. (("person", False),), (("n_restarts", 1),)
    boost: Tuple[Tuple[str, object], ...] = ()    # booster overrides, e.g. (("monotone", False),), (("additive", True),)
    boost_off: bool = False             # skip stage 2 (U = stage-1 log-probabilities)
    shuffle_sentences: bool = False     # control: members see outcome sentences of a random other event of the same alternative
    sentence_control: str = ""          # "" | "random" (E ~ N(0,1)) | "altid" (E = one-hot alternative identity)
    cold_start: bool = False            # drop the training rows of persons who appear in val/test (no panel)
    no_hist: bool = False               # zero the history features (is_repeat, purchase_count) everywhere


@dataclass
class StructuralStage:
    model: nn.Module                    # fitted hetero_v1 ensemble (full-train fit)
    logits: torch.Tensor                # (N, J) L from the full-train fit
    offset: torch.Tensor                # (N, J) training offset: cross-fitted on train rows, = logits elsewhere
    info: Dict


_STRUCT_CACHE: Dict[Tuple[str, int, int], StructuralStage] = {}
_MEMBER_CACHE: Dict[Tuple[str, int, int], Tuple[List[PrefBranch], Dict]] = {}


# ----------------------------------------------------------------------------- stage 1
def _fold_ids(b: Bundle, folds: int, by_person: bool, seed: int) -> torch.Tensor:
    """Fold id per train row; grouped by person on cold-start data, by event where the panel exists."""
    tr = b.idx("train")
    g = torch.Generator().manual_seed(seed)
    if not by_person:
        return torch.randperm(len(tr), generator=g) % folds
    persons = b.person[tr]
    uniq = persons.unique()
    fold_of = torch.zeros(int(uniq.max()) + 1, dtype=torch.long)
    fold_of[uniq[torch.randperm(len(uniq), generator=g)]] = torch.arange(len(uniq)) % folds
    return fold_of[persons]


def _cross_fit(b: Bundle, seed: int, folds: int, lam: Optional[float], wd: float,
               by_person: bool, struct: Tuple[Tuple[str, object], ...] = ()) -> Tuple[torch.Tensor, Dict]:
    """Stage-1 log-probabilities on train rows from fold-complement fits (one restart, the
    selected person shrinkage and weight decay); zeros elsewhere."""
    tr = b.idx("train")
    fold = _fold_ids(b, folds, by_person, seed)
    form = {**HETERO_V1, **{k: v for k, v in dict(struct).items() if k not in ("n_restarts", "net_wd")}}
    builder = make_hetero_builder(**form, n_restarts=1, net_wd=wd,
                                  person_l2_grid=(lam,) if lam is not None else (0.0,))
    out = torch.zeros(b.N, b.J)
    fold_val = []
    for k in range(folds):
        split = b.split.clone()
        split[tr[fold == k]] = 3                                        # held out of every split
        fb = dataclasses.replace(b, split=split)
        model, run = builder(fb, seed)
        info = run(model, fb, seed)
        hold = tr[fold == k]
        with torch.no_grad():
            model.eval()
            out[hold] = F.log_softmax(model(fb, hold), 1)
        fold_val.append(info["extra"]["stage1_val_nll"])
    return out, {"folds": folds, "by_person": by_person, "fold_val_nll": fold_val}


def structural_stage(b: Bundle, seed: int, oof_folds: int,
                     struct: Tuple[Tuple[str, object], ...] = (), view: str = "") -> StructuralStage:
    key = (b.dataset, seed, oof_folds, struct, view)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    t0 = time.time()
    kw = {**HETERO_V1, "n_restarts": 5, "net_wd": 1e-2, **dict(struct)}
    model, run = (build_hetero_v1 if not struct else make_hetero_builder(**kw))(b, seed)
    info = run(model, b, seed)
    ex = info["extra"]
    logits = F.log_softmax(all_logits(model, b), 1)
    offset = logits.clone()
    oof_info: Dict = {"folds": 0}
    if oof_folds > 0:
        oof, oof_info = _cross_fit(b, seed, oof_folds, ex["person_l2"], ex["net_wd"], by_person=not ex["person_effects"],
                                   struct=struct)
        tr = b.idx("train")
        offset[tr] = oof[tr]
    va, tr = b.idx("val"), b.idx("train")
    summary = {"stage1_val_nll": ex["stage1_val_nll"], "stage1_train_nll_insample": float(F.cross_entropy(logits[tr], b.y[tr])),
               "stage1_train_nll_offset": float(F.cross_entropy(offset[tr], b.y[tr])),
               "person_effects": ex["person_effects"], "person_l2": ex["person_l2"], "net_wd": ex["net_wd"],
               "panel_fraction_val": ex["panel_fraction_val"], "vot": ex["vot"], "person": ex["person"],
               "cross_fit": oof_info, "seconds": time.time() - t0}
    _STRUCT_CACHE[key] = StructuralStage(model=model, logits=logits, offset=offset, info=summary)
    return _STRUCT_CACHE[key]


# ----------------------------------------------------------------------------- stage 1c
class ConceptResidual(nn.Module):
    """offset + g . C(E, x, z): the ncat_v3 concept branch trained log-linearly on the structural
    residual (cross-fitted offset on train rows in training mode, full-fit logits otherwise)."""

    def __init__(self, b: Bundle, offset: torch.Tensor, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("offset", offset)
        self.register_buffer("logits", logits)
        self.branch = ConceptBranch(b, **NCAT_V3)
        self.g = nn.Parameter(torch.tensor(1.0))

    def term(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return self.g * self.branch(b, idx)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        base = self.offset[idx] if self.training else self.logits[idx]
        return base + self.term(b, idx)


def concept_stage(b: Bundle, seed: int, stage1: StructuralStage) -> Tuple[torch.Tensor, Dict]:
    """Fit the concept residual; return its utility term on all rows (N, J) and fit info."""
    torch.manual_seed(seed)
    m = ConceptResidual(b, stage1.offset, stage1.logits)
    tk = dict(NCAT_TRAIN)
    groups = [{"params": list(m.branch.parameters())}, {"params": [m.g], "lr": tk["lr"], "weight_decay": 0.0}]
    fr = fit(m, b, params=groups, seed=seed, **tk)
    m.eval()
    with torch.no_grad():
        idx = torch.arange(b.N)
        term = torch.cat([m.term(b, idx[s:s + 4096]) for s in range(0, b.N, 4096)], 0)
    te = b.idx("test")
    spread = float((term[te] - term[te].mean(1, keepdim=True)).std())
    return term, {"gate": float(m.g), "best_epoch": fr.best_epoch, "best_val_nll": fr.best_val_nll,
                  "offset_val_nll": fr.val_curve[0], "term_std_test": spread, "seconds": fr.seconds,
                  "n_params": int(sum(p.numel() for p in m.branch.parameters()))}


# ----------------------------------------------------------------------------- stage 2
def boost_stage(b: Bundle, seed: int, init: torch.Tensor,
                boost: Tuple[Tuple[str, object], ...] = ()) -> Tuple[torch.Tensor, Dict]:
    """RUM-shaped boosted residual with ``init`` (N, J, record slot order) as the offset.
    Returns the residual utilities f (N, J, slot order) at the early-stopped round and fit info."""
    init_c = _canon(b, init).numpy().astype(np.float64)                 # (N, A)
    y_c = b.alt_idx[torch.arange(b.N), b.y].numpy()
    blocks, _ = build_blocks(b, None, seed)
    splits = {"tr": b.idx("train").numpy(), "va": b.idx("val").numpy(), "te": b.idx("test").numpy()}
    payload = {}
    for s, ix in splits.items():
        payload[f"y_{s}"], payload[f"init_{s}"] = y_c[ix], init_c[ix]
    for a, (X, _, mono) in enumerate(blocks):
        payload[f"mono{a}"] = mono
        for s, ix in splits.items():
            payload[f"X{a}_{s}"] = X[ix]
    ccfg = {"mode": "rum", "A": b.n_alts, "seed": seed, "use_init": True,
            **{k: BOOST_CFG[k] for k in ("lr", "num_leaves", "min_data", "lambda_l2", "feature_fraction",
                                         "bagging_fraction", "patience", "max_rounds", "num_threads", "additive", "monotone")}}
    ccfg.update(dict(boost))
    if not ccfg["monotone"]:
        for a in range(b.n_alts):
            payload[f"mono{a}"] = np.zeros_like(payload[f"mono{a}"])
    out = run_child(payload, ccfg)
    f_c = torch.zeros(b.N, b.n_alts, dtype=torch.float64)
    for s, ix in splits.items():
        f_c[torch.as_tensor(ix)] = torch.as_tensor(out[f"U_{s}"] - init_c[ix])
    imps = {}
    for a, (_, names, _) in enumerate(blocks):
        imp = out[f"importance{a}"]
        tot = imp.sum() or 1.0
        imps[b.meta["alts"][a]] = {names[i]: round(float(imp[i] / tot), 3) for i in np.argsort(-imp)[:8] if imp[i] > 0}
    info = {"best_round": int(out["best_round"]), "n_trees": int(out["n_trees"]), "best_val_nll": float(out["best_val_nll"]),
            "importance": imps}
    return f_c.float().gather(1, b.alt_idx), info


# ----------------------------------------------------------------------------- controls
class _ShuffledE:
    """Lazy view of E where slot (i, j) returns the sentences of a random other event's slot that holds
    the same canonical alternative and lies in the same split (train / val / test)."""

    def __init__(self, E: torch.Tensor, src_i: torch.Tensor, src_j: torch.Tensor):
        self._E, self._si, self._sj = E, src_i, src_j
        self.shape = E.shape

    def __getitem__(self, idx):
        return self._E[self._si[idx], self._sj[idx]]


class _ConstE:
    """Lazy E view with no event-specific content: ``random`` draws a fixed N(0,1) tensor once
    (seeded), ``altid`` returns the one-hot canonical alternative identity padded to d dims."""

    def __init__(self, b: Bundle, kind: str, seed: int):
        self.shape = b.E.shape
        self.kind, self.alt_idx, self.d, self.K = kind, b.alt_idx, b.d, b.K
        if kind == "random":
            g = torch.Generator().manual_seed(seed + 4242)
            self._E = torch.randn(b.N, b.J, b.K, b.d, generator=g, dtype=torch.float16)
        elif kind == "altid":
            self.A = b.n_alts
        else:
            raise ValueError(kind)

    def __getitem__(self, idx):
        if self.kind == "random":
            return self._E[idx].float()
        oh = F.one_hot(self.alt_idx[idx], self.A).float()                          # (n, J, A)
        return F.pad(oh, (0, self.d - self.A))[:, :, None, :].expand(-1, -1, self.K, -1)


def control_view(b: Bundle, kind: str, seed: int) -> Bundle:
    return dataclasses.replace(b, E=_ConstE(b, kind, seed))


def cold_start_view(b: Bundle) -> Bundle:
    """Training rows of persons who also appear in val or test are held out of every split."""
    tr, held = b.idx("train"), torch.zeros(b.N, dtype=torch.bool)
    seen = torch.zeros(b.n_persons, dtype=torch.bool)
    seen[b.person[torch.cat([b.idx("val"), b.idx("test")])]] = True
    held[tr[seen[b.person[tr]]]] = True
    split = b.split.clone(); split[held] = 3
    return dataclasses.replace(b, split=split)


def no_hist_view(b: Bundle) -> Bundle:
    return dataclasses.replace(b, Xhist=torch.zeros_like(b.Xhist))


def shuffled_view(b: Bundle, seed: int) -> Bundle:
    g = torch.Generator().manual_seed(seed + 977)
    src_i = torch.arange(b.N)[:, None].expand(b.N, b.J).clone()
    src_j = torch.arange(b.J)[None, :].expand(b.N, b.J).clone()
    for split in (0, 1, 2):
        for a in range(b.n_alts):
            pos = torch.nonzero((b.alt_idx == a) & (b.split[:, None] == split), as_tuple=False)   # (m, 2)
            perm = pos[torch.randperm(len(pos), generator=g)]
            src_i[pos[:, 0], pos[:, 1]] = perm[:, 0]
            src_j[pos[:, 0], pos[:, 1]] = perm[:, 1]
    return dataclasses.replace(b, E=_ShuffledE(b.E, src_i, src_j))


# ----------------------------------------------------------------------------- stage 3
def _pretrain_member(m: PrefBranch, b: Bundle, seed: int):
    probe = lambda m_, b_, s_: PREF_LAM_PROBE * F.cross_entropy(m.probe_logits(b_, s_), b_.y[s_])
    return fit(m, b, params=list(m.parameters()), seed=seed, extra_loss=probe, **PREF_TRAIN)


def semantic_members(b: Bundle, seed: int, members: int, shuffled: bool = False,
                     view: str = "") -> Tuple[List[PrefBranch], Dict]:
    key = (b.dataset, seed, members, shuffled, view)
    if key in _MEMBER_CACHE:
        return _MEMBER_CACHE[key]
    torch.manual_seed(seed)
    mems = [PrefBranch(b, **PREF_V1) for _ in range(members)]
    frs = [_pretrain_member(m, b, seed * 100 + i) for i, m in enumerate(mems)]
    ens = _SemEnsemble(mems)
    sem_test = {k: evaluate(ens, b, "test")[k] for k in ("top1", "nll", "brier", "ece")}
    info = {"member_best_val": [f.best_val_nll for f in frs], "member_best_epoch": [f.best_epoch for f in frs],
            "sem_only_test": sem_test, "seconds": sum(f.seconds for f in frs),
            "n_params": int(sum(p.numel() for m in mems for p in m.parameters()))}
    _MEMBER_CACHE[key] = (mems, info)
    return _MEMBER_CACHE[key]


class _SemEnsemble(nn.Module):
    def __init__(self, members: Sequence[PrefBranch]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        return _log_mean_exp(torch.stack([m(b, idx) for m in self.members], 1), 1)


# ----------------------------------------------------------------------------- the model
class Omleu2(nn.Module):
    """log p_i(j) = logaddexp(log(1 - pi) + log_softmax(a U_i)_j, log pi + log mean_m p_m^sem(j)).

    ``U`` holds the boosted structural utilities (stage 2) for every row; ``structural`` and
    ``concept`` keep the fitted stage-1 / 1c modules (parameter count, interpretation)."""

    def __init__(self, b: Bundle, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("U", torch.zeros(b.N, b.J))
        self.structural: Optional[nn.Module] = None
        self.members = nn.ModuleList()
        self.gamma = nn.Parameter(torch.tensor(-6.0))                    # pi = sigmoid(gamma) ~ 0.0025 at init
        self.log_a = nn.Parameter(torch.zeros(()))
        self.sem_view: Optional[Bundle] = None                            # sentence-control / data view for the members
        self.data_view: Optional[Bundle] = None                           # cold-start / no-history view (informational)

    @property
    def pi(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        a = self.log_a.exp() if self.cfg.temp else 1.0
        Lp = torch.log_softmax(a * self.U[idx], -1)
        if len(self.members) == 0:
            return Lp
        bb = self.sem_view if self.sem_view is not None else b
        Sp = _log_mean_exp(torch.stack([m(bb, idx) for m in self.members], 1), 1)
        return torch.logaddexp(torch.log1p(-self.pi) + Lp, torch.log(self.pi) + Sp)

    def stack_on_val(self, b: Bundle) -> float:
        """Fit (gamma, log_a) on the validation split by L-BFGS; returns the validation NLL."""
        scalars = ([self.gamma] if len(self.members) else []) + ([self.log_a] if self.cfg.temp else [])
        va = b.idx("val")
        self.eval()
        if not scalars:
            return nll_on(self, b, va)
        return lbfgs_prefit(lambda: F.cross_entropy(self(b, va), b.y[va]), scalars, max_iter=200)


def _build(b: Bundle, seed: int, cfg: Config):
    model = Omleu2(b, cfg)

    def run(model: Omleu2, b: Bundle, seed: int) -> Dict:
        t0 = time.time()
        view = ("cold," if cfg.cold_start else "") + ("nohist," if cfg.no_hist else "")
        if cfg.cold_start:
            b = cold_start_view(b)
        if cfg.no_hist:
            b = no_hist_view(b)
        model.data_view = b if view else None
        va = b.idx("val")
        s1 = structural_stage(b, seed, cfg.oof_folds, cfg.struct, view)
        model.structural = s1.model
        logits, offset = s1.logits, s1.offset
        info: Dict = {"stage1": s1.info}
        if cfg.ncat:
            term, ncat_info = concept_stage(b, seed, s1)
            logits, offset = logits + term, offset + term
            info["stage1c"] = ncat_info
        trials = {}
        best: Tuple[float, float, torch.Tensor, Dict] = (float("inf"), 0.0, torch.zeros(b.N, b.J), {})
        tau_grid = (1.0,) if cfg.boost_off else cfg.tau_grid
        for tau in tau_grid:
            if cfg.boost_off:
                f, binfo = torch.zeros(b.N, b.J), {"best_val_nll": float(F.cross_entropy(logits[va], b.y[va])),
                                                   "best_round": 0, "n_trees": 0, "importance": {}}
            else:
                f, binfo = boost_stage(b, seed, tau * offset, cfg.boost)
            trials[str(tau)] = {"val_nll": binfo["best_val_nll"], "best_round": binfo["best_round"]}
            if binfo["best_val_nll"] < best[0]:
                best = (binfo["best_val_nll"], tau, f, binfo)
        tau = best[1]
        model.U.copy_(tau * logits + best[2])
        info["stage2"] = {"tau": tau, "trials": trials, "init_val_nll": float(F.cross_entropy(tau * logits[va], b.y[va])),
                          **best[3]}
        if cfg.members:
            sb = shuffled_view(b, seed) if cfg.shuffle_sentences else b
            if cfg.sentence_control:
                sb = control_view(sb, cfg.sentence_control, seed)
            mems, minfo = semantic_members(sb, seed, cfg.members, shuffled=cfg.shuffle_sentences,
                                           view=view + cfg.sentence_control)
            model.members = nn.ModuleList(mems)
            model.sem_view = sb if (cfg.shuffle_sentences or cfg.sentence_control or view) else None
            info["stage3"] = minfo
        val_nll = model.stack_on_val(b)
        extra = {**info, "cfg": dataclasses.asdict(cfg), "tau": tau, "pi": float(model.pi), "temp_a": float(model.log_a.exp()),
                 "gate": float(model.pi)}
        return {"fit": {"best_epoch": best[3]["best_round"], "best_val_nll": val_nll, "boost_val_nll": best[0],
                        "stage1_val_nll": s1.info["stage1_val_nll"], "epochs": 0, "seconds": time.time() - t0},
                "extra": extra}
    return model, run


def make_builder(**kw):
    cfg = Config(**kw)
    return lambda b, seed: _build(b, seed, cfg)


build_combo_struct = make_builder()
build_combo_struct_cal = make_builder(temp=True)
build_combo_full = make_builder(members=5, temp=True)
build_combo_full_ncat = make_builder(members=5, temp=True, ncat=True)
build_combo_struct_ncat = make_builder(ncat=True)
build_combo_struct_insample = make_builder(oof_folds=0)

# Ablations of the full model (combo_full minus one component; see docs/ablation_plan.md)
FULL = dict(members=5, temp=True)
build_abl_no_person = make_builder(**FULL, struct=(("person", False),))
build_abl_no_taste = make_builder(**FULL, struct=(("taste", False),))
build_abl_no_intercepts = make_builder(**FULL, struct=(("intercepts", False),))
build_abl_linear_struct = make_builder(**FULL, struct=(("person", False), ("taste", False), ("intercepts", False)))
build_abl_restarts1 = make_builder(**FULL, struct=(("n_restarts", 1),))
build_abl_no_boost = make_builder(**FULL, boost_off=True)
build_abl_boost_nomono = make_builder(**FULL, boost=(("monotone", False),))
build_abl_boost_additive = make_builder(**FULL, boost=(("additive", True),))
build_abl_tau1_insample = make_builder(**FULL, oof_folds=0, tau_grid=(1.0,))
build_abl_members1 = make_builder(members=1, temp=True)
build_abl_shuffled_sentences = make_builder(**FULL, shuffle_sentences=True)
build_abl_random_embeddings = make_builder(**FULL, sentence_control="random")
build_abl_altid_sentences = make_builder(**FULL, sentence_control="altid")
build_abl_cold_start = make_builder(**FULL, cold_start=True)
build_abl_cold_start_struct = make_builder(cold_start=True)
build_abl_no_hist = make_builder(**FULL, no_hist=True)
