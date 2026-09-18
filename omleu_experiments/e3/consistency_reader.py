"""E3 training: the axis reader trained with the consistency losses.

The losses need three views of the same events: the original sentences, a claim-preserving
paraphrase, and the sentences re-rendered after a controlled edit.  All three exist for the
deterministic template channel, so this reader trains on templates; the same code runs on
generated text as soon as a generation budget allows the edited and paraphrased sentences to
be produced.

Paraphrase families and edit magnitudes are split: the families and magnitudes used during
training are disjoint from the ones used to score behavioural consistency, and every derived
view of an event stays in that event's own partition because the views are built row-wise
from the same bundle.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from experiments.harness.data import Bundle
from omleu_experiments.contracts import AXES
from omleu_experiments.e2.axis_reader import AxisNet, AxisReader
from omleu_experiments.views import replace_embeddings

from .edits import EDITS, edited_embeddings
from .losses import collapse_diagnostics, local_loss, order_loss, paraphrase_loss
from .paraphrase import paraphrase_embeddings

TRAIN_FAMILIES = ("recorded_to_measured", "colon_to_clause")
HELD_OUT_FAMILIES = ("passive", "notrecorded")
TRAIN_EDITS = ("fare_up", "transfer_up")
HELD_OUT_EDITS = ("wait_up", "access_up")


class ConsistencyAxisReader(AxisReader):
    """Axis reader plus paraphrase, ordering and locality losses."""

    name = "consistency_axis"

    def __init__(self, *, lam_para: float = 0.3, lam_order: float = 0.3, lam_local: float = 0.1,
                 edit_slot: int = 0, cache_dir=None, **kw):
        super().__init__(**kw)
        self.lam_para, self.lam_order, self.lam_local = lam_para, lam_order, lam_local
        self.edit_slot, self.cache_dir = edit_slot, cache_dir
        self._views: Dict[str, Bundle] = {}
        self._axis_index = {a: i for i, a in enumerate(AXES)}

    def _prepare(self, b: Bundle) -> None:
        if self._views:
            return
        fam = TRAIN_FAMILIES[0]
        self._views["para"] = replace_embeddings(b, paraphrase_embeddings(b, fam, cache_dir=self.cache_dir))
        spec = EDITS[TRAIN_EDITS[0]]
        E, elig = edited_embeddings(b, spec, self.edit_slot, cache_dir=self.cache_dir)
        self._views["edit"] = replace_embeddings(b, E)
        self._spec = spec
        self._eligible = torch.as_tensor(elig)

    def _aux_loss(self, m: AxisNet):
        lam_p, lam_o, lam_l = self.lam_para, self.lam_order, self.lam_local
        slot = self.edit_slot

        def f(_model, b: Bundle, sel: torch.Tensor):
            self._prepare(b)
            loss = b.y.new_zeros((), dtype=torch.float32)
            s0 = m.scores(b, sel)
            if lam_p > 0:
                loss = loss + lam_p * paraphrase_loss(s0, m.scores(self._views["para"], sel))
            if lam_o > 0 or lam_l > 0:
                keep = self._eligible[sel]
                if bool(keep.any()):
                    sub = sel[keep]
                    a = [self._axis_index[x] for x in self._spec.affected_axes if x in self._axis_index]
                    u = [self._axis_index[x] for x in self._spec.unaffected_axes if x in self._axis_index]
                    so, se = m.scores(b, sub), m.scores(self._views["edit"], sub)
                    if lam_o > 0 and a:
                        loss = loss + lam_o * order_loss(so, se, a, slot)
                    if lam_l > 0 and u:
                        loss = loss + lam_l * local_loss(so, se, u, slot)
            return loss
        return f

    def behavioural_report(self, b: Bundle, rows: Sequence[int]) -> Dict[str, float]:
        """Held-out paraphrase families and edit magnitudes, plus collapse diagnostics."""
        idx = torch.as_tensor(np.asarray(rows), dtype=torch.long)
        out: Dict[str, float] = {}
        m = self.members[0]
        with torch.no_grad():
            s0 = m.scores(b, idx)
            out.update({f"collapse_{k}": v for k, v in collapse_diagnostics(s0).items()})
            for fam in HELD_OUT_FAMILIES:
                bp = replace_embeddings(b, paraphrase_embeddings(b, fam, cache_dir=self.cache_dir))
                sp = m.scores(bp, idx)
                out[f"para_drift_{fam}"] = float((s0 - sp).pow(2).mean().sqrt())
                out[f"para_prob_divergence_{fam}"] = float(
                    (m(b, idx).exp() - m(bp, idx).exp()).abs().sum(-1).mean())
            for name in HELD_OUT_EDITS:
                spec = EDITS[name]
                if spec.feature_index(b) is None:
                    continue
                E, elig = edited_embeddings(b, spec, self.edit_slot, cache_dir=self.cache_dir)
                keep = torch.as_tensor(elig)[idx]
                if not bool(keep.any()):
                    continue
                sub = idx[keep]
                se = m.scores(replace_embeddings(b, E), sub)
                so = m.scores(b, sub)
                a = [self._axis_index[x] for x in spec.affected_axes if x in self._axis_index]
                u = [self._axis_index[x] for x in spec.unaffected_axes if x in self._axis_index]
                d = (se[:, self.edit_slot, a] - so[:, self.edit_slot, a])
                out[f"order_violation_rate_{name}"] = float((d > 1e-6).float().mean())
                out[f"order_violation_magnitude_{name}"] = float(torch.clamp(d, min=0).mean())
                out[f"spillover_rms_{name}"] = float(
                    (se[:, self.edit_slot, u] - so[:, self.edit_slot, u]).pow(2).mean().sqrt()) if u else float("nan")
                out[f"n_eligible_{name}"] = int(keep.sum())
        return out
