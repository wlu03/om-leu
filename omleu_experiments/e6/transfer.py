"""E6: transfer of the outcome scorer between datasets.

What transfers is the map from an axis sentence to an outcome score: the projection and the
per-axis heads.  What does not transfer is the valuation, because the covariate vectors,
the alternatives and the units differ; it is refitted on the target, and so is the
calibration.  Alternatives need not overlap: the scorer is alternative-agnostic and the
availability mask handles the rest.

Units are not silently equated.  The recorded quantities enter the text with their own unit
words (CHF, GBP, minutes), and score scale is normalised on the *source* only, so the target
never rescales the frozen scorer using its own labels.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from experiments.harness.data import Bundle, load_bundle
from omleu_experiments.e2.axis_reader import AxisNet, AxisReader
from omleu_experiments.readers import SemanticReader
from omleu_experiments.views import fold_view, inner_split, restandardise_covariates

# Axes are identical across the three datasets by construction (the same five prompt axes).
# Attributes differ in name and unit; the mapping below is used only for the deterministic
# template channel, and is explicit rather than assumed.
SCHEMA_MAP = {
    "financial": {"swissmetro": ("cost_100chf", "100 CHF"), "optima": ("cost_chf", "CHF"), "lpmc": ("cost_gbp", "GBP")},
    "time": {"swissmetro": ("time_h", "hours"), "optima": ("time_h", "hours"), "lpmc": ("time_h", "hours")},
    "convenience": {"swissmetro": (None, None), "optima": ("walking_h", "hours"), "lpmc": ("access_h", "hours")},
    "reliability": {"swissmetro": ("headway_h", "hours"), "optima": ("waiting_h", "hours"), "lpmc": (None, None)},
    "comfort": {"swissmetro": ("seats", "class"), "optima": ("distance_km", "km"), "lpmc": (None, None)},
}


class TransferReader(SemanticReader):
    """Fit an axis scorer on ``source``, freeze it, refit only the valuation on the target."""

    name = "transfer"

    def __init__(self, *, source: str, source_seed: int = 7, sentence_source: str = "llm",
                 freeze: bool = True, axis_kw: Optional[Dict] = None, **kw):
        super().__init__(**kw)
        self.source, self.source_seed, self.freeze = source, source_seed, freeze
        self.axis_kw = dict(axis_kw or {})
        self.src_sentence_source = sentence_source
        self._source_state: Optional[List[Dict]] = None

    # ---- source ---------------------------------------------------------------------
    def _fit_source(self, seed: int) -> List[Dict]:
        from omleu_experiments.runner import cluster_ids, person_strings
        from omleu_experiments.protocol import person_disjoint
        from omleu_experiments.sentences import build_source
        from omleu_experiments.views import replace_embeddings
        b = load_bundle(self.source, self.source_seed)
        persons, _, _ = person_strings(b)
        cl = cluster_ids(b, persons)
        part = person_disjoint(persons, cl, self.source_seed, n_folds=2)
        E = build_source(b, self.src_sentence_source, seed=self.source_seed)
        bs = replace_embeddings(b, E)
        f_rows, v_rows = inner_split(part.train_dev, cl, self.source_seed)
        bv = restandardise_covariates(fold_view(bs, f_rows, v_rows, part.test))
        r = AxisReader(**self.axis_kw); r.n_members = self.n_members
        r.fit(bv, self.source_seed)
        self.info = {"source": self.source, "source_seed": self.source_seed,
                     "source_events_used": int(len(f_rows)),
                     "source_clusters_used": int(len(set(cl[f_rows].tolist()))),
                     "source_member_val": r.info.get("best_val")}
        return [copy.deepcopy(m.state_dict()) for m in r.members]

    # ---- target ---------------------------------------------------------------------
    def _member(self, b: Bundle) -> nn.Module:
        return AxisNet(b, **self.axis_kw)

    def fit(self, b: Bundle, seed: int) -> None:
        if self._source_state is None:
            self._source_state = self._fit_source(seed)
        torch.manual_seed(seed)
        self.members = []
        scorer_keys = ("proj.", "norm.", "heads.")
        for i in range(self.n_members):
            m = self._member(b)
            src = self._source_state[i % len(self._source_state)]
            transferable = {k: v for k, v in src.items()
                            if k.startswith(scorer_keys) and k in m.state_dict()
                            and m.state_dict()[k].shape == v.shape}
            m.load_state_dict(transferable, strict=False)
            if self.freeze:
                for name, p in m.named_parameters():
                    if name.startswith(scorer_keys):
                        p.requires_grad_(False)
            self.members.append(m)
        from experiments.harness.train import fit as _fit
        fits = [_fit(m, b, params=[p for p in m.parameters() if p.requires_grad], seed=seed * 100 + i,
                     lr=1e-3, max_epochs=80, patience=8, batch_size=256, weight_decay=1e-3)
                for i, m in enumerate(self.members)]
        n_frozen = sum(p.numel() for p in self.members[0].parameters() if not p.requires_grad)
        n_train = sum(p.numel() for p in self.members[0].parameters() if p.requires_grad)
        self.info = {**self.info, "transferred_tensors": len(transferable), "frozen_parameters": int(n_frozen),
                     "trainable_parameters": int(n_train), "target_best_val": [f.best_val_nll for f in fits],
                     "freeze": self.freeze}
