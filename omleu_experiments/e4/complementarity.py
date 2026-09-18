"""E4: train the semantic channel against what the numeric channel already predicts.

Independent pretraining is preserved: every member is first fitted on its own choice loss.
Fine-tuning then minimises

    L_mix = -mean log( (1 - pi_train) p_numeric[y] + pi_train q_sem[y] )

with ``p_numeric`` **detached** and produced out of fold *inside the outer training
partition*, so the semantic channel is never fine-tuned against numeric predictions that
saw those events' labels.  ``pi_train`` is an interior constant (0.2 by default) and is not
the final calibrated ``pi``; the final calibration uses held-out predictions of the whole
fine-tuned procedure.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.harness.data import Bundle
from experiments.harness.train import fit
from omleu_experiments.numeric import fit_numeric
from omleu_experiments.readers import READERS, SemanticReader
from omleu_experiments.views import inner_split


def mixture_aware_loss(q_logprobs: torch.Tensor, p_numeric_logprobs: torch.Tensor, y: torch.Tensor,
                       pi_train: float) -> torch.Tensor:
    """Log-space mixture NLL with the numeric side treated as a constant."""
    lp = torch.logaddexp(math.log(1.0 - pi_train) + p_numeric_logprobs.detach(),
                         math.log(pi_train) + q_logprobs)
    return -lp[torch.arange(len(y)), y].mean()


def inner_numeric_logprobs(b: Bundle, seed: int, view_tag: str, *, person_effects: bool,
                           inner_folds: int = 3) -> torch.Tensor:
    """Out-of-fold numeric log-probabilities for every row of this view's training partition.

    Rows outside the training partition keep ``-log J`` and are never used by the loss.
    """
    tr = b.idx("train").numpy()
    cl = np.asarray([str(int(p)) for p in b.person.numpy()])
    out = torch.full((b.N, b.J), -math.log(b.J), dtype=torch.float64)
    clusters = np.array(sorted(set(cl[tr].tolist())))
    rng = np.random.default_rng(seed)
    fold_of = {c: i % inner_folds for i, c in enumerate(clusters[rng.permutation(len(clusters))])}
    assign = np.array([fold_of[c] for c in cl[tr]])
    for k in range(inner_folds):
        hold = tr[assign == k]
        rest = tr[assign != k]
        if len(hold) == 0 or len(rest) < 50:
            continue
        f_rows, v_rows = inner_split(rest, cl, seed * 7 + k)
        nf = fit_numeric(b, fit_rows=f_rows, val_rows=v_rows, predict_rows=hold, seed=seed,
                         view_tag=f"{view_tag}in{k}", person_effects=person_effects)
        u = torch.as_tensor(nf.logits, dtype=torch.float64)
        out[torch.as_tensor(hold, dtype=torch.long)] = torch.log_softmax(u, -1)
    return out


class MixtureAwareReader(SemanticReader):
    """Wraps any registered reader: pretrain independently, then fine-tune on ``L_mix``."""

    name = "mixture_aware"

    def __init__(self, *, base: str = "preserved", base_kw: Optional[Dict] = None, pi_train: float = 0.2,
                 lam_standalone: float = 0.5, epochs: int = 25, lr: float = 3e-4, inner_folds: int = 3,
                 person_effects: bool = False, **kw):
        super().__init__(**kw)
        self.base_name, self.base_kw = base, dict(base_kw or {})
        self.pi_train, self.lam_standalone = pi_train, lam_standalone
        self.epochs, self.lr, self.inner_folds = epochs, lr, inner_folds
        self.person_effects = person_effects
        self._base: Optional[SemanticReader] = None

    def fit(self, b: Bundle, seed: int) -> None:
        base_cls = READERS[self.base_name]
        self._base = base_cls(**self.base_kw)
        self._base.n_members = self.n_members
        self._base.fit(b, seed)                                   # independent pretraining, preserved
        P = inner_numeric_logprobs(b, seed, f"e4s{seed}", person_effects=self.person_effects,
                                   inner_folds=self.inner_folds)
        tr, va = b.idx("train"), b.idx("val")
        assert torch.isfinite(P[tr]).all(), "inner numeric predictions are not finite"
        assert float(P[tr].exp().sum(-1).max()) <= 1.0 + 1e-6, "inner numeric rows are not probabilities"
        self.members = self._base.members
        hist = []
        for m in self.members:
            opt = torch.optim.Adam(m.parameters(), lr=self.lr, weight_decay=1e-3)
            best, best_state, bad = math.inf, copy.deepcopy(m.state_dict()), 0
            for ep in range(self.epochs):
                m.train()
                perm = tr[torch.randperm(len(tr))]
                for s in range(0, len(perm), 256):
                    sel = perm[s:s + 256]
                    opt.zero_grad()
                    q = m(b, sel).double()
                    loss = mixture_aware_loss(q, P[sel], b.y[sel], self.pi_train)
                    if self.lam_standalone > 0:
                        loss = loss + self.lam_standalone * F.cross_entropy(m(b, sel), b.y[sel])
                    loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    v = float(mixture_aware_loss(m(b, va).double(), P[va], b.y[va], self.pi_train))
                if v < best - 1e-5:
                    best, best_state, bad = v, copy.deepcopy(m.state_dict()), 0
                else:
                    bad += 1
                    if bad >= 5:
                        break
            m.load_state_dict(best_state); m.eval()
            hist.append(best)
        self.info = {**self._base.info, "mixture_aware": {"pi_train": self.pi_train, "val_mix_nll": hist,
                                                          "inner_folds": self.inner_folds,
                                                          "lam_standalone": self.lam_standalone}}
