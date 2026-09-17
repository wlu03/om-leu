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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import evaluate, fit, lbfgs_prefit, nll_on
from experiments.models.boost import DEFAULT as BOOST_DEFAULT, RUM, _canon, all_logits, build_blocks, run_child
from experiments.models.hetero import build_hetero_v1, make_builder as make_hetero_builder
from experiments.models.ncat import ConceptBranch
from experiments.models.orthogonal import ResidualMember, erased_view, probe_r2
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
    pi_input: str = ""                  # "" scalar pi | "Z": covariate gate | "S": uncertainty gate on [H_struct, H_sem, member disagreement]
    pi_fit: str = "val"                 # "val" L-BFGS on validation | "boot": bootstrap-median on validation | "oof": stacked on
                                        # person-grouped out-of-fold predictions over train+val (Wolpert / super-learner stacking)
    oof_pi_folds: int = 5
    pi_boot: int = 100
    pi_l2: float = 1.0                  # ridge on w (per validation event) for the covariate gate
    slot_keep: Tuple[int, ...] = ()     # members see only these sentence slots (others zeroed); () = all K
    erase: bool = False                 # erase [numeric attributes, alternative identity, Z] from E before the members
    resid_nu: float = 0.0               # > 0: members trained with nu * cross-fitted structural log-probs as an offset
    member_kind: str = "pref"           # stage-3 member class (MEMBER_FACTORIES key); "pref" = PrefBranch V1
    member_kw: Tuple[Tuple[str, object], ...] = ()   # keyword overrides for the member class (ablations/)


class _LogAClamp:
    """Smooth bound on the inverse temperature: a = exp(lo + (hi-lo) * (tanh(x)+1)/2).

    A hard clamp has zero gradient outside the range, which stalls L-BFGS; this keeps the
    gradient finite everywhere while holding a in [e^lo, e^hi].  The range covers every
    temperature the datasets here have needed (fitted values run from 0.7 to 1.3)."""

    def __init__(self, lo: float = -4.0, hi: float = 4.0):
        self.lo, self.hi = lo, hi

    def tanh_scale(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.lo + (self.hi - self.lo) * 0.5 * (torch.tanh(x) + 1.0))

    def inverse(self, a: float) -> float:
        import math
        z = (math.log(a) - self.lo) / (self.hi - self.lo) * 2.0 - 1.0
        return math.atanh(min(max(z, -0.999999), 0.999999))


LOG_A_CLAMP = _LogAClamp()


@dataclass
class StructuralStage:
    model: nn.Module                    # fitted hetero_v1 ensemble (full-train fit)
    logits: torch.Tensor                # (N, J) L from the full-train fit
    offset: torch.Tensor                # (N, J) training offset: cross-fitted on train rows, = logits elsewhere
    info: Dict


_ERASE_CACHE: Dict[Tuple[str, int, str], Tuple[Bundle, Dict]] = {}
_STRUCT_CACHE: Dict[Tuple[str, int, int], StructuralStage] = {}


def erased_stage(b: Bundle, seed: int, view: str) -> Tuple[Bundle, Dict]:
    """Erased-E view of ``b`` (fitted on its own training rows) with the probe R2 before and after.

    Every direction linearly predictable from the alternative's numeric attributes, its identity and
    the person covariates is removed from each sentence embedding, so whatever the language channel
    contributes afterwards cannot be a linear restatement of those inputs."""
    key = (b.dataset, seed, view)
    if key not in _ERASE_CACHE:
        t0 = time.time()
        eb = erased_view(b)
        info = {"probe_before": probe_r2(b, b.E), "probe_after": probe_r2(b, eb.E), "seconds": time.time() - t0}
        _ERASE_CACHE[key] = (eb, info)
    return _ERASE_CACHE[key]
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


class _SlotE:
    """Lazy E view that zeroes every sentence slot except ``keep`` (per-axis ablation)."""

    def __init__(self, E, keep: Tuple[int, ...]):
        self._E, self.shape = E, E.shape
        m = torch.zeros(E.shape[2]); m[list(keep)] = 1.0
        self._mask = m[None, None, :, None]

    def __getitem__(self, idx):
        return self._E[idx] * self._mask


def slot_view(b: Bundle, keep: Tuple[int, ...]) -> Bundle:
    return dataclasses.replace(b, E=_SlotE(b.E, keep))


def cold_start_view(b: Bundle, seed: int) -> Bundle:
    """Person-level re-split (70 / 15 / 15 of persons, seeded): no person appears in two splits.
    The test set differs from the paired protocol, so these runs are only comparable with each other."""
    g = torch.Generator().manual_seed(seed + 31)
    persons = torch.randperm(b.n_persons, generator=g)
    n_val = n_te = int(round(0.15 * b.n_persons))
    part = torch.zeros(b.n_persons, dtype=torch.int8)
    part[persons[:n_te]] = 2
    part[persons[n_te:n_te + n_val]] = 1
    return dataclasses.replace(b, split=part[b.person])


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
# member classes: name -> factory(bundle, **kw) -> nn.Module with forward(b, idx) -> log-probs (n, J)
# and probe_logits(b, idx) -> logits or None.  ablation/ registers plain networks here.
MEMBER_FACTORIES: Dict[str, Callable] = {"pref": lambda b, **kw: PrefBranch(b, **{**PREF_V1, **kw})}


def _pretrain_member(m: nn.Module, b: Bundle, seed: int):
    has_probe = getattr(m, "has_probe", True)
    probe = (lambda m_, b_, s_: PREF_LAM_PROBE * F.cross_entropy(m.probe_logits(b_, s_), b_.y[s_])) if has_probe else None
    return fit(m, b, params=list(m.parameters()), seed=seed, extra_loss=probe, **PREF_TRAIN)


def semantic_members(b: Bundle, seed: int, members: int, shuffled: bool = False,
                     view: str = "", kind: str = "pref", kw: Tuple[Tuple[str, object], ...] = (),
                     offset: Optional[torch.Tensor] = None, nu: float = 0.0) -> Tuple[List[nn.Module], Dict]:
    key = (b.dataset, seed, members, shuffled, view, kind, kw, nu)
    if key in _MEMBER_CACHE:
        return _MEMBER_CACHE[key]
    torch.manual_seed(seed)
    mems = [MEMBER_FACTORIES[kind](b, **dict(kw)) for _ in range(members)]
    if nu > 0:
        mems = [ResidualMember(m, offset, nu) for m in mems]
    frs = [_pretrain_member(m, b, seed * 100 + i) for i, m in enumerate(mems)]
    ens = _SemEnsemble(mems)
    ev = evaluate(ens, b, "test")
    sem_test = {k: ev[k] for k in ("top1", "nll", "brier", "ece")}
    info = {"member_best_val": [f.best_val_nll for f in frs], "member_best_epoch": [f.best_epoch for f in frs],
            "sem_only_test": sem_test, "sem_only_per_event_nll": ev["per_event_nll"], "seconds": sum(f.seconds for f in frs),
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
        self.w_pi = nn.Parameter(torch.zeros({"Z": b.P, "S": 3}.get(cfg.pi_input, 0))) if cfg.pi_input in ("Z", "S") else None
        self._sfeat_stats = None
        self.sem_view: Optional[Bundle] = None                            # sentence-control / data view for the members
        self.data_view: Optional[Bundle] = None                           # cold-start / no-history view (informational)

    @property
    def pi(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma.clamp(-30.0, 30.0))

    @property
    def a(self) -> torch.Tensor:
        """Inverse temperature actually applied to the numeric utilities."""
        return LOG_A_CLAMP.tanh_scale(self.log_a) if self.cfg.temp else torch.ones(())

    def unc_features(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        """[structural entropy, semantic entropy, member disagreement] per event, standardised on val."""
        bb = self.sem_view if self.sem_view is not None else b
        with torch.no_grad():
            Lp = torch.log_softmax(self.U[idx], -1)
            M = torch.stack([m(bb, idx) for m in self.members], 1)               # (n, M, J)
            Sp = _log_mean_exp(M, 1)
            h_s = -(Lp.exp() * Lp).sum(-1)
            h_m = -(Sp.exp() * Sp).sum(-1)
            dis = M.exp().std(1).mean(-1)                                        # disagreement of member probabilities
            f = torch.stack([h_s, h_m, dis], 1)
        if self._sfeat_stats is None:
            self._sfeat_stats = (f.mean(0), f.std(0) + 1e-6)
        mu, sd = self._sfeat_stats
        return (f - mu) / sd

    def pi_of(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        """Mixture weight per event: scalar, or a logistic function of covariates / uncertainty features."""
        if self.w_pi is None:
            return self.pi.expand(len(idx))
        feats = b.Z[idx] if self.cfg.pi_input == "Z" else self.unc_features(b, idx)
        return torch.sigmoid(self.gamma + feats @ self.w_pi)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        a = self.a
        Lp = torch.log_softmax(a * self.U[idx], -1)
        if len(self.members) == 0:
            return Lp
        bb = self.sem_view if self.sem_view is not None else b
        Sp = _log_mean_exp(torch.stack([m(bb, idx) for m in self.members], 1), 1)
        pi = self.pi_of(b, idx)[:, None]
        return torch.logaddexp(torch.log1p(-pi) + Lp, torch.log(pi) + Sp)

    def stack_on_val(self, b: Bundle) -> float:
        """Fit (gamma, log_a) on the validation split by L-BFGS; returns the validation NLL."""
        scalars = ([self.gamma] if len(self.members) else []) + ([self.log_a] if self.cfg.temp else [])
        if self.w_pi is not None and len(self.members):
            scalars.append(self.w_pi)
        va = b.idx("val")
        self.eval()
        if not scalars:
            return nll_on(self, b, va)
        l2 = self.cfg.pi_l2
        if self.cfg.pi_input == "S":
            self._sfeat_stats = None
            self.unc_features(b, va)                                             # fix standardisation on validation

        def loss(sel=va):
            l = F.cross_entropy(self(b, sel), b.y[sel])
            if self.w_pi is not None:
                l = l + l2 * (self.w_pi ** 2).sum() / len(sel)
            return l
        if self.cfg.pi_fit == "boot":
            # bootstrap the validation events, refit the scalars each time, keep the median: a
            # small-sample shrinkage against pi overshooting on a few hundred validation events
            g = torch.Generator().manual_seed(1234)
            fits = []
            for _ in range(self.cfg.pi_boot):
                sel = va[torch.randint(0, len(va), (len(va),), generator=g)]
                for p_ in scalars:
                    p_.data.zero_()
                self.gamma.data.fill_(-6.0)
                lbfgs_prefit(lambda: loss(sel), scalars, max_iter=100)
                fits.append([p_.detach().clone() for p_ in scalars])
            for k, p_ in enumerate(scalars):
                p_.data.copy_(torch.stack([f[k] for f in fits]).median(0).values)
            return nll_on(self, b, va)
        lbfgs_prefit(loss, scalars, max_iter=200)
        return nll_on(self, b, va)


def _fit_scalars_oof(model: "Omleu2", b: Bundle, seed: int, cfg: Config, view: str) -> Dict:
    """Stack pi (and the temperature) on out-of-fold predictions: split the persons of train+val into
    K folds; for each fold run the full pipeline (pi_fit='val') on the complement with that fold as its
    validation split, and record the fold's structural log-probabilities and semantic log-probabilities.
    Then fit the scalars by L-BFGS on the pooled out-of-fold predictions (K times more persons than a
    single validation split) and copy them into ``model`` (whose stages were fitted on the full train)."""
    K = cfg.oof_pi_folds
    pool = torch.cat([b.idx("train"), b.idx("val")])
    persons = b.person[pool].unique()
    g = torch.Generator().manual_seed(seed + 7)
    if cfg.cold_start:
        # cold-start regime: folds must hold out whole persons so the fold matches the test regime
        fold_of = torch.zeros(int(persons.max()) + 1, dtype=torch.long)
        fold_of[persons[torch.randperm(len(persons), generator=g)]] = torch.arange(len(persons)) % K
        fold_row = fold_of[b.person[pool]]
    else:
        # warm (chronological) regime: hold out events, keeping every person's other events in
        # the fold's training data, so the fold matches a test split that shares respondents
        fold_row = torch.randperm(len(pool), generator=g) % K
    sub_cfg = dataclasses.replace(cfg, pi_fit="val", cold_start=False, no_hist=False)   # views already applied to b
    Ls, Ss, ys = [], [], []
    for k in range(K):
        split = b.split.clone()
        held = pool[fold_row == k]
        split[pool] = 0
        split[held] = 1
        fb = dataclasses.replace(b, split=split)
        m_k, run_k = _build(fb, seed, sub_cfg, view_tag=f"{view}oof{k}")
        run_k(m_k, fb, seed)
        idx = fb.idx("val")
        with torch.no_grad():
            Lp = torch.log_softmax(m_k.U[idx], -1)
            bb = m_k.sem_view if m_k.sem_view is not None else fb
            Sp = _log_mean_exp(torch.stack([m(bb, idx) for m in m_k.members], 1), 1)
        Ls.append(Lp); Ss.append(Sp); ys.append(fb.y[idx])
    L, S, y = torch.cat(Ls), torch.cat(Ss), torch.cat(ys)
    gamma, log_a = model.gamma, model.log_a
    gamma.data.fill_(-6.0); log_a.data.zero_()

    def loss():
        # Bound both scalars inside the objective.  Without this the line search can push
        # log_a or gamma far enough that exp() or log() overflows on a channel whose
        # log-probabilities are extreme (residual-trained members, or a very sharp
        # structural channel), and the fit dies instead of converging.
        a = LOG_A_CLAMP.tanh_scale(log_a) if cfg.temp else 1.0
        Lp = torch.log_softmax(a * L, -1)
        pi = torch.sigmoid(gamma.clamp(-30.0, 30.0))
        lp = torch.logaddexp(torch.log1p(-pi + 1e-12) + Lp, torch.log(pi + 1e-12) + S)
        return F.cross_entropy(lp, y)
    scalars = [gamma] + ([log_a] if cfg.temp else [])
    nll = lbfgs_prefit(loss, scalars, max_iter=200)
    return {"folds": K, "n_oof_events": int(len(y)), "n_oof_persons": int(len(persons)), "oof_nll_at_fit": float(loss()),
            "oof_nll_struct_only": float(F.cross_entropy(L, y)), "oof_nll_sem_only": float(F.cross_entropy(S, y))}


def _build(b: Bundle, seed: int, cfg: Config, view_tag: str = ""):
    model = Omleu2(b, cfg)

    def run(model: Omleu2, b: Bundle, seed: int) -> Dict:
        t0 = time.time()
        view = ("cold," if cfg.cold_start else "") + ("nohist," if cfg.no_hist else "") + view_tag
        if cfg.cold_start:
            b = cold_start_view(b, seed)
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
            sb = b
            if cfg.erase:            # erase first, then shuffle: the control must not leak the numbers
                sb, einfo = erased_stage(b, seed, view)
                info["erase"] = {"probe_before": einfo["probe_before"], "probe_after": einfo["probe_after"]}
            if cfg.shuffle_sentences:
                sb = shuffled_view(sb, seed)
            if cfg.sentence_control:
                sb = control_view(sb, cfg.sentence_control, seed)
            if cfg.slot_keep:
                sb = slot_view(sb, cfg.slot_keep)
            mems, minfo = semantic_members(sb, seed, cfg.members, shuffled=cfg.shuffle_sentences,
                                           view=view + cfg.sentence_control
                                                + (f"slots{cfg.slot_keep}" if cfg.slot_keep else "")
                                                + ("erase" if cfg.erase else ""),
                                           kind=cfg.member_kind, kw=cfg.member_kw,
                                           offset=s1.offset if cfg.resid_nu > 0 else None, nu=cfg.resid_nu)
            model.members = nn.ModuleList(mems)
            model.sem_view = sb if (cfg.shuffle_sentences or cfg.sentence_control or view
                                    or cfg.slot_keep or cfg.erase) else None
            info["stage3"] = minfo
        if cfg.pi_fit == "oof" and cfg.members:
            oof_info = _fit_scalars_oof(model, b, seed, cfg, view)
            info["oof_stacking"] = oof_info
            val_nll = nll_on(model, b, b.idx("val"))
        else:
            val_nll = model.stack_on_val(b)
        extra = {**info, "cfg": dataclasses.asdict(cfg), "tau": tau, "pi": float(model.pi), "temp_a": float(model.a),
                 "gate": float(model.pi)}
        if model.w_pi is not None:
            with torch.no_grad():
                pv = model.pi_of(b, b.idx("test"))
            extra["pi_test"] = {"mean": float(pv.mean()), "p10": float(pv.quantile(0.1)), "p90": float(pv.quantile(0.9))}
            extra["pi_weights"] = {n: float(w) for n, w in sorted(zip(b.meta["z_names"], model.w_pi.detach()), key=lambda t: -abs(t[1]))[:8]}
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
build_abl_cold_start = make_builder(**FULL, cold_start=True, struct=(("person", False),))
build_abl_cold_start_struct = make_builder(cold_start=True, struct=(("person", False),))
build_abl_no_hist = make_builder(**FULL, no_hist=True)
# person-level re-split protocol: the within-protocol ladder MNL -> MNL + trees -> hetero + trees -> + semantic
LIN = (("person", False), ("taste", False), ("intercepts", False))
build_abl_cold_start_linear = make_builder(cold_start=True, struct=LIN, boost_off=True)
build_abl_cold_start_linear_boost = make_builder(cold_start=True, struct=LIN)
build_abl_cold_start_linear_full = make_builder(**FULL, cold_start=True, struct=LIN)



# --- LLM-signal experiments (docs/llm_signal_research.md) ---
COLD = dict(cold_start=True, struct=(("person", False),))
build_llm_cold_shuffled = make_builder(**FULL, **COLD, shuffle_sentences=True)
build_llm_cold_altid = make_builder(**FULL, **COLD, sentence_control="altid")
build_llm_cold_random = make_builder(**FULL, **COLD, sentence_control="random")
build_llm_gate_z = make_builder(**FULL, pi_input="Z")
build_llm_cold_gate_z = make_builder(**FULL, **COLD, pi_input="Z")
build_llm_gate_z_strong = make_builder(**FULL, pi_input="Z", pi_l2=20.0)
# --- pi-estimation robustness and uncertainty gate (why control gains != realised gains) ---
build_pi_boot = make_builder(**FULL, pi_fit="boot")
build_pi_cold_boot = make_builder(**FULL, **COLD, pi_fit="boot")
build_pi_unc_gate = make_builder(**FULL, pi_input="S", pi_l2=5.0)
build_pi_cold_unc_gate = make_builder(**FULL, **COLD, pi_input="S", pi_l2=5.0)
build_pi_cold_unc_gate_boot = make_builder(**FULL, **COLD, pi_input="S", pi_l2=5.0, pi_fit="boot")
build_pi_oof = make_builder(**FULL, pi_fit="oof")
build_pi_cold_oof = make_builder(**FULL, **COLD, pi_fit="oof")
build_llm_cold_gate_z_strong = make_builder(**FULL, **COLD, pi_input="Z", pi_l2=20.0)
for _k, _name in enumerate(("financial", "time", "comfort", "convenience", "reliability")):
    globals()[f"build_llm_slot_{_name}"] = make_builder(**FULL, slot_keep=(_k,))
    globals()[f"build_llm_cold_slot_{_name}"] = make_builder(**FULL, **COLD, slot_keep=(_k,))

# ---------------------------------------------------------------------------------------------
# Identified outcome channel: the contribution is certified non-redundant with the observed
# attributes (erasure), or trained against what the structural model already predicts (residual
# offset), and the mixture weight is stacked out of fold rather than set by hand.  Each variant
# ships with its own shuffled control, which shuffles the *already erased* embeddings so the
# control cannot recover the numbers through the subtracted term.
# ---------------------------------------------------------------------------------------------
ID_COLD = dict(**FULL, **COLD, pi_fit="oof")
build_id_erase = make_builder(**ID_COLD, erase=True)
build_id_erase_shuffled = make_builder(**ID_COLD, erase=True, shuffle_sentences=True)
build_id_resid = make_builder(**ID_COLD, resid_nu=0.3)
build_id_resid_shuffled = make_builder(**ID_COLD, resid_nu=0.3, shuffle_sentences=True)
build_id_erase_resid = make_builder(**ID_COLD, erase=True, resid_nu=0.3)
build_id_erase_resid_shuffled = make_builder(**ID_COLD, erase=True, resid_nu=0.3, shuffle_sentences=True)
build_id_plain = make_builder(**ID_COLD)
build_id_plain_shuffled = make_builder(**ID_COLD, shuffle_sentences=True)
