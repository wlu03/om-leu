"""Heterogeneous, panel-aware structural part for the hybrid model.

Structural utility of alternative j for event i (person p, covariates z):

    V_ij = ASC_j + beta . X_lin_ij                      linear MNL columns (other LOS, z x alt, history)
         + beta_time,j(z) . time_ij + beta_cost,j(z) . cost_ij   TasteNet: -softplus(r_j + delta_j(z)) < 0
         + g_j(z)                                       functional intercept (small MLP, zero-init)
         + u_pj                                         person random effect (L2-shrunk embedding)

optionally passed through a nested-logit probability layer with learned nest
scales, and optionally plus ``gate * SemanticBranch`` (stage 2, as in
``hybrid_base``).

Estimation: stage 1a L-BFGS on the linear/sign-constrained coefficients (all
nets output exactly zero at init), stage 1b Adam on the nets + person effects
(+ linear coefficients) with early stopping on validation NLL and the person
shrinkage selected on validation, stage 2 (optional) Adam on the semantic
branch behind a zero-initialised gate.

Person effects are only enabled where the panel exists (val persons seen in
train); on cold-start splits (Optima) they are switched off automatically,
which is also what "unseen persons get zero" would give at test time.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import fit, lbfgs_prefit, nll_on
from experiments.models.baseline import SemanticBranch

COST_UNIT = {"swissmetro": 100.0, "optima": 1.0, "lpmc": 1.0}   # multiply VOT to get currency / hour
CURRENCY = {"swissmetro": "CHF/h", "optima": "CHF/h", "lpmc": "GBP/h"}


def zero_mlp(n_in: int, hidden: int, n_out: int, dropout: float = 0.1) -> nn.Sequential:
    """Small MLP whose last layer is zero-initialised (outputs exactly 0 at init)."""
    m = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))
    nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


class HeteroStructural(nn.Module):
    def __init__(self, b: Bundle, *, taste="alt", intercepts: bool = True, person: bool = True,
                 nested: bool = False, semantic: bool = False, with_z_linear: bool = True,
                 taste_hidden: int = 16, int_hidden: int = 32, dropout: float = 0.1, z_clip: float = 4.0,
                 sem_kw: Optional[dict] = None):
        super().__init__()
        taste = {True: "alt", False: False}.get(taste, taste)   # False | "alt" | "generic"
        self.cfg = dict(taste=taste, intercepts=intercepts, person=person, nested=nested, semantic=semantic,
                        with_z_linear=with_z_linear, taste_hidden=taste_hidden, int_hidden=int_hidden)
        A = b.n_alts
        self.A = A
        tname = b.meta["alt_feature_names"][b.meta["time_idx"]]
        cname = b.meta["alt_feature_names"][b.meta["cost_idx"]]
        X, names = b.mnl_columns(with_z=with_z_linear, with_hist=True)
        if taste:
            keep = [i for i, n in enumerate(names) if not (n.startswith(f"los_{tname}@") or n.startswith(f"los_{cname}@"))]
            X, names = X[:, :, keep], [names[i] for i in keep]
        self.register_buffer("X", X.contiguous())
        self.names = names
        self.beta = nn.Parameter(torch.zeros(X.shape[-1]))
        self.register_buffer("Zc", b.Z.clamp(-z_clip, z_clip))
        self.register_buffer("alt_idx", b.alt_idx)
        self.register_buffer("person_idx", b.person)

        # --- TasteNet time / cost coefficients, per canonical alternative ---
        self.taste = bool(taste)
        self.generic = taste == "generic"
        if taste:
            oh = b.alt_onehot()
            tr = b.idx("train")
            if self.generic:                                        # one coefficient shared by all alts: (N, J, 1)
                oh = torch.ones_like(oh[:, :, :1]); A = 1
            T = b.Xnum[:, :, b.meta["time_idx"]][:, :, None] * oh   # (N, J, A) raw hours
            C = b.Xnum[:, :, b.meta["cost_idx"]][:, :, None] * oh
            s_t = T[tr].reshape(-1, A).std(0); s_c = C[tr].reshape(-1, A).std(0)
            self.register_buffer("time_valid", (s_t > 1e-6).float())
            self.register_buffer("cost_valid", (s_c > 1e-6).float())
            s_t = torch.where(s_t > 1e-6, s_t, torch.ones_like(s_t)); s_c = torch.where(s_c > 1e-6, s_c, torch.ones_like(s_c))
            self.register_buffer("s_time", s_t); self.register_buffer("s_cost", s_c)
            self.register_buffer("T", (T / s_t).contiguous()); self.register_buffer("C", (C / s_c).contiguous())
            self.r_time = nn.Parameter(torch.zeros(A)); self.r_cost = nn.Parameter(torch.zeros(A))
            self.taste_net = zero_mlp(b.P, taste_hidden, 2 * A, dropout)
            self.A_taste = A
            A = self.A
        # --- functional intercepts ---
        self.intercepts = intercepts
        if intercepts:
            self.int_net = zero_mlp(b.P, int_hidden, A, dropout)
        # --- person random effects ---
        self.person = person
        if person:
            self.pe = nn.Embedding(b.n_persons, A); nn.init.zeros_(self.pe.weight)
            cnt = torch.bincount(b.person[b.idx("train")], minlength=b.n_persons).float()
            self.register_buffer("person_count", cnt)
        # --- nested logit ---
        self.nested = nested
        if nested:
            nests = list(b.meta["nests"].items())
            nest_of = torch.zeros(A, dtype=torch.long)
            for k, (nm, members) in enumerate(nests):
                for a in members:
                    nest_of[a] = k
            self.nest_names = [nm for nm, _ in nests]
            self.register_buffer("nest_of", nest_of)
            self.register_buffer("nest_free", torch.tensor([len(m) > 1 for _, m in nests], dtype=torch.float32))
            self.nest_raw = nn.Parameter(torch.full((len(nests),), -4.0))   # lambda = exp(-softplus(raw)) ~ 0.98
        # --- semantic branch ---
        self.semantic = semantic
        if semantic:
            self.sem = SemanticBranch(b, **(sem_kw or {}))
            self.gate = nn.Parameter(torch.tensor(0.0))

    # ---- pieces ----
    def linear_params(self) -> List[nn.Parameter]:
        ps = [self.beta]
        if self.taste:
            ps += [self.r_time, self.r_cost]
        if self.nested:
            ps.append(self.nest_raw)
        return ps

    def net_params(self) -> List[nn.Parameter]:
        ps = []
        if self.taste:
            ps += list(self.taste_net.parameters())
        if self.intercepts:
            ps += list(self.int_net.parameters())
        return ps

    def nest_lambda(self) -> torch.Tensor:
        lam = torch.exp(-F.softplus(self.nest_raw))
        return lam * self.nest_free + (1.0 - self.nest_free)

    def taste_coefs(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled-unit coefficients (n, A) for time and cost, both < 0."""
        d = self.taste_net(self.Zc[idx])                        # (n, 2A)
        At = self.A_taste
        bt = -F.softplus(self.r_time + d[:, :At]) * self.time_valid
        bc = -F.softplus(self.r_cost + d[:, At:]) * self.cost_valid
        return bt, bc

    def structural(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        V = (self.X[idx] * self.beta).sum(-1)                   # (n, J)
        alt = self.alt_idx[idx]                                 # (n, J)
        if self.taste:
            bt, bc = self.taste_coefs(idx)                      # (n, A)
            V = V + (self.T[idx] * bt[:, None, :]).sum(-1) + (self.C[idx] * bc[:, None, :]).sum(-1)
        if self.intercepts:
            g = self.int_net(self.Zc[idx])                      # (n, A)
            g = g - g[:, :1]                                    # reference alternative 0
            V = V + torch.gather(g, 1, alt)
        if self.person:
            u = self.pe(self.person_idx[idx])                   # (n, A)
            V = V + torch.gather(u, 1, alt)
        return V

    def nested_logprob(self, V: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        alt = self.alt_idx[idx]
        m = self.nest_of[alt]                                   # (n, J) nest id per position
        lam = self.nest_lambda()                                # (n_nests,)
        lam_pos = lam[m]                                        # (n, J)
        a = V / lam_pos
        ivs = []
        for k in range(len(lam)):
            mask = (m == k)
            ivs.append(torch.logsumexp(a.masked_fill(~mask, -1e9), dim=1))
        IV = torch.stack(ivs, 1)                                # (n, n_nests)
        lower = a - torch.gather(IV, 1, m)                      # log P(j | nest)
        upper = lam[None, :] * IV                               # (n, n_nests)
        upper = upper - torch.logsumexp(upper, 1, keepdim=True)
        return lower + torch.gather(upper, 1, m)

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        V = self.structural(b, idx)
        if self.semantic:
            V = V + self.gate * self.sem(b, idx)
        if self.nested:
            return self.nested_logprob(V, idx)
        return V

    # ---- estimation ----
    def prefit(self, b: Bundle, l2: float = 1e-4) -> float:
        tr = b.idx("train")
        self.eval()

        def loss():
            return F.cross_entropy(self.forward(b, tr), b.y[tr]) + l2 * (self.beta ** 2).sum()
        return lbfgs_prefit(loss, self.linear_params())

    def person_penalty(self, lam: float):
        """lam / N_train * sum_p ||u_p||^2, spread over the events of each person."""
        def pen(model, b, sel):
            p = model.person_idx[sel]
            u = model.pe(p)
            return lam * ((u ** 2).sum(1) / model.person_count[p].clamp_min(1.0)).mean()
        return pen

    # ---- reporting ----
    @torch.no_grad()
    def vot_summary(self, b: Bundle) -> Dict:
        if not self.taste:
            return {}
        self.eval()
        te = b.idx("test")
        unit = COST_UNIT.get(b.dataset, 1.0)
        bt, bc = self.taste_coefs(te)
        bt_raw = bt / self.s_time; bc_raw = bc / self.s_cost      # per hour, per currency unit
        bt0 = -F.softplus(self.r_time) * self.time_valid / self.s_time
        bc0 = -F.softplus(self.r_cost) * self.cost_valid / self.s_cost
        out = {"currency": CURRENCY.get(b.dataset, "?"), "generic": self.generic}
        for a, an in enumerate(["generic"] if self.generic else b.meta["alts"]):
            rec = {"beta_time_per_h_mean": float(bt_raw[:, a].mean()), "beta_cost_mean": float(bc_raw[:, a].mean()),
                   "beta_time_sd": float(bt_raw[:, a].std()), "beta_cost_sd": float(bc_raw[:, a].std())}
            if self.time_valid[a] > 0 and self.cost_valid[a] > 0:
                vot = (bt_raw[:, a] / bc_raw[:, a].clamp(max=-1e-8)) * unit
                q = torch.quantile(vot, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
                rec.update({"vot_mean": float(vot.mean()), "vot_p10": float(q[0]), "vot_p25": float(q[1]),
                            "vot_median": float(q[2]), "vot_p75": float(q[3]), "vot_p90": float(q[4]),
                            "vot_homogeneous_r": float(bt0[a] / bc0[a].clamp(max=-1e-8) * unit)})
            out[an] = rec
        return out

    @torch.no_grad()
    def person_summary(self, b: Bundle) -> Dict:
        if not self.person:
            return {"enabled": False}
        u = self.pe.weight[self.person_count > 0]
        return {"enabled": True, "n_persons_train": int((self.person_count > 0).sum()),
                "effect_sd": float(u.std()), "effect_abs_mean": float(u.abs().mean()),
                "effect_max_abs": float(u.abs().max())}


def _panel_fraction(b: Bundle) -> float:
    tr = set(b.person[b.idx("train")].tolist()); va = b.person[b.idx("val")].tolist()
    return sum(p in tr for p in va) / max(1, len(va))


class Ensemble(nn.Module):
    """Average of member probabilities (restarts of stage 1b); optional semantic branch on top."""

    def __init__(self, members: List[HeteroStructural], b: Bundle, semantic: bool = False, sem_kw=None):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.semantic = semantic
        if semantic:
            self.sem = SemanticBranch(b, **(sem_kw or {}))
            self.gate = nn.Parameter(torch.tensor(0.0))

    def structural_logp(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        lps = torch.stack([F.log_softmax(m(b, idx), 1) for m in self.members], 0)
        return torch.logsumexp(lps, 0) - math.log(len(self.members))

    def forward(self, b: Bundle, idx: torch.Tensor) -> torch.Tensor:
        lp = self.structural_logp(b, idx)
        if self.semantic:
            lp = lp + self.gate * self.sem(b, idx)
        return lp


def make_builder(*, taste="alt", intercepts=True, person=True, nested=False, semantic=False,
                 person_l2_grid=(0.03, 0.1, 0.3, 1.0, 3.0), net_wd=1e-3, lr=1e-3, lin_lr=1e-3, pe_lr=None,
                 max_epochs=300, patience=20, taste_hidden=16, int_hidden=32, dropout=0.1,
                 n_restarts=1, sem_kw=None):
    def build(b: Bundle, seed: int):
        panel = _panel_fraction(b)
        use_person = person and panel >= 0.5
        torch.manual_seed(seed)
        base = HeteroStructural(b, taste=taste, intercepts=intercepts, person=use_person, nested=nested,
                                semantic=False, taste_hidden=taste_hidden, int_hidden=int_hidden, dropout=dropout)
        model = Ensemble([base], b, semantic=semantic, sem_kw=sem_kw) if (n_restarts > 1 or semantic) else base

        wd_grid = list(net_wd) if isinstance(net_wd, (tuple, list)) else [net_wd]

        def stage1b(m: HeteroStructural, lam: float, seed: int, wd: float):
            groups = [{"params": m.linear_params(), "lr": lin_lr, "weight_decay": 0.0}]
            if m.net_params():
                groups.append({"params": m.net_params(), "lr": lr, "weight_decay": wd})
            if m.person:
                groups.append({"params": [m.pe.weight], "lr": pe_lr or lr, "weight_decay": 0.0})
            extra = m.person_penalty(lam) if m.person else None
            return fit(m, b, params=groups, lr=lr, max_epochs=max_epochs, patience=patience, batch_size=256,
                       seed=seed, extra_loss=extra)

        def run(model, b: Bundle, seed: int):
            m0 = base
            ce1a = m0.prefit(b)
            val1a = nll_on(m0, b, b.idx("val"))
            info: Dict = {"stage1a_train_ce": ce1a, "stage1a_val_nll": val1a, "panel_fraction_val": panel,
                          "person_effects": use_person, "n_restarts": n_restarts}
            base_state = copy.deepcopy(m0.state_dict())
            grid = list(person_l2_grid) if use_person else [None]
            wds = wd_grid if m0.net_params() else [wd_grid[0]]
            best = (math.inf, None, None, None, None)
            trials = {}
            for wd in wds:
                for lam in grid:
                    m0.load_state_dict(base_state)
                    fr = stage1b(m0, lam if lam is not None else 0.0, seed, wd)
                    trials[f"l2={lam},wd={wd}"] = {"val": fr.best_val_nll, "best_epoch": fr.best_epoch, "epochs": fr.epochs}
                    if fr.best_val_nll < best[0]:
                        best = (fr.best_val_nll, lam, copy.deepcopy(m0.state_dict()), fr, wd)
            m0.load_state_dict(best[2])
            fr1 = best[3]
            lam, wd = best[1], best[4]
            info.update({"stage1b": {k: v for k, v in fr1.__dict__.items() if k != "val_curve"},
                         "person_l2": lam, "net_wd": wd, "stage1b_trials": trials, "stage1_val_nll_single": fr1.best_val_nll})
            secs = fr1.seconds
            # restarts (different init / dropout / batch order), averaged in probability
            if n_restarts > 1:
                for r in range(1, n_restarts):
                    mr = copy.deepcopy(m0); mr.load_state_dict(base_state)
                    torch.manual_seed(seed * 1000 + r)
                    for mod in (mr.taste_net if mr.taste else None, mr.int_net if mr.intercepts else None):
                        if mod is not None:
                            mod[0].reset_parameters()
                    frr = stage1b(mr, lam if lam is not None else 0.0, seed * 1000 + r, wd)
                    secs += frr.seconds
                    model.members.append(mr)
                info["stage1_val_nll"] = nll_on(model, b, b.idx("val")) if not semantic else None
            if isinstance(model, Ensemble):
                model.semantic, sem_flag = False, model.semantic
                info["stage1_val_nll"] = nll_on(model, b, b.idx("val"))
                model.semantic = sem_flag
            else:
                info["stage1_val_nll"] = fr1.best_val_nll
            fit_out = {**fr1.__dict__, "seconds": secs}
            if semantic:
                sem_params = list(model.sem.parameters()) + [model.gate]
                fr2 = fit(model, b, params=sem_params, lr=1e-3, max_epochs=60, patience=8, batch_size=256, seed=seed)
                info.update({"stage2": {k: v for k, v in fr2.__dict__.items() if k != "val_curve"},
                             "gate": float(model.gate), "stage2_val_nll": fr2.best_val_nll})
                fit_out = {**fr2.__dict__, "seconds": secs + fr2.seconds}
            members = model.members if isinstance(model, Ensemble) else [model]
            vot = members[0].vot_summary(b)
            if len(members) > 1 and vot:
                for an in [k for k, v in vot.items() if isinstance(v, dict)]:
                    meds = [mm.vot_summary(b)[an].get("vot_median") for mm in members]
                    meds = [x for x in meds if x is not None]
                    if meds:
                        vot[an]["vot_median_restart_mean"] = float(np.mean(meds)); vot[an]["vot_median_restart_sd"] = float(np.std(meds))
            extra = {**info, "vot": vot, "person": members[0].person_summary(b), "cfg": members[0].cfg,
                     "hparams": dict(net_wd=net_wd, lr=lr, lin_lr=lin_lr, pe_lr=pe_lr or lr, dropout=dropout, taste_hidden=taste_hidden,
                                     int_hidden=int_hidden, max_epochs=max_epochs, patience=patience)}
            if nested:
                extra["nest_lambda"] = dict(zip(members[0].nest_names, members[0].nest_lambda().tolist()))
            return {"fit": {k: v for k, v in fit_out.items() if k != "val_curve"}, "extra": extra}
        return model, run
    return build


# ---- registered variants ----
# hetero_v1: generic TasteNet beta_time(z), beta_cost(z) (< 0) + functional intercepts g_j(z) + person random
#            effects (shrinkage selected on val), fixed net weight decay 1e-2, 5 restarts averaged in probability.
# hetero_v2: alternative-specific TasteNet coefficients (per-alt VOT), net weight decay selected on val
#            from {1e-3, 1e-2}, otherwise as v1.
# Both use plain MNL softmax; pass semantic=True for the OM-LEU semantic branch behind a zero-init gate
# (stage 2), nested=True for the nested-logit layer.
build_hetero_v1 = make_builder(taste="generic", intercepts=True, person=True, n_restarts=5, net_wd=1e-2)
build_hetero_v2 = make_builder(taste="alt", intercepts=True, person=True, n_restarts=5, net_wd=(1e-3, 1e-2))

# Explored but not registered (scratch names used in the development log). Register by adding
# ("experiments.models.hetero", "EXPLORED['<name>']")-style entries or a thin wrapper if needed.
EXPLORED = {
    "lin": make_builder(taste=False, intercepts=False, person=False),          # linear MNL + Adam polish (= mnl_only)
    "pe": make_builder(taste=False, intercepts=False, person=True),
    "pe5": make_builder(taste=False, intercepts=False, person=True, n_restarts=5),
    "taste": make_builder(taste="alt", intercepts=False, person=False),
    "gtaste": make_builder(taste="generic", intercepts=False, person=False),
    "int": make_builder(taste=False, intercepts=True, person=False),
    "v1_single": make_builder(taste="alt", intercepts=True, person=True),       # one restart, wd 1e-3
    "v3_nested": make_builder(taste="alt", intercepts=True, person=True, nested=True),
    "v4_generic_single": make_builder(taste="generic", intercepts=True, person=True),
    "v6_alt_wd1e-2": make_builder(taste="alt", intercepts=True, person=True, n_restarts=5, net_wd=1e-2),
    "v7_semantic": make_builder(taste="generic", intercepts=True, person=True, n_restarts=5, net_wd=1e-2, semantic=True),
    "v8_generic_wdsel": make_builder(taste="generic", intercepts=True, person=True, n_restarts=5, net_wd=(1e-3, 1e-2)),
    "v10_semantic_wdsel": make_builder(taste="generic", intercepts=True, person=True, n_restarts=5, net_wd=(1e-3, 1e-2), semantic=True),
    "v11_fast_pe": make_builder(taste="generic", intercepts=True, person=True, n_restarts=5, net_wd=(1e-3, 1e-2),
                                pe_lr=1e-2, person_l2_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
    "v14_nested": make_builder(taste="generic", intercepts=True, person=True, n_restarts=5, net_wd=(1e-3, 1e-2),
                               nested=True, pe_lr=1e-2, person_l2_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0)),
}
