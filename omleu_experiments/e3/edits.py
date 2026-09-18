"""Controlled edits to recorded attributes.

An edit changes one physical quantity by a stated amount in stated units, keeps logically
linked fields coherent, and declares which axes it may change and which it must not.  The
edited event has **no choice label**: nobody observed this person facing the edited
alternative, so an edit is a behavioural restriction on the representation, not a
counterfactual observation, and it identifies nothing causally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.harness.data import Bundle
from omleu_experiments.contracts import AXES


@dataclass(frozen=True)
class EditSpec:
    name: str
    feature_candidates: Tuple[str, ...]   # first one present in the dataset is edited
    delta: float                          # additive change in the dataset's own units
    units: str
    affected_axes: Tuple[str, ...]        # axes whose score may change
    unaffected_axes: Tuple[str, ...]      # axes that must not change
    direction: int                        # -1: the edited alternative becomes no better on affected axes
    linked: Tuple[Tuple[str, float], ...] = ()   # other features kept coherent, (name, delta)
    eligibility: str = "attribute must be recorded and positive after the edit"

    def feature_index(self, b: Bundle) -> Optional[int]:
        names = b.meta["alt_feature_names"]
        for c in self.feature_candidates:
            if c in names:
                return names.index(c)
        return None


EDITS: Dict[str, EditSpec] = {
    "fare_up": EditSpec("fare_up", ("cost_chf", "cost_gbp", "cost_100chf"), +2.0, "dataset cost unit",
                        ("financial",), ("time", "reliability"), -1,
                        eligibility="service attributes held fixed; cost must be recorded"),
    "wait_up": EditSpec("wait_up", ("waiting_h", "headway_h"), +0.25, "hours",
                        ("reliability", "time"), ("financial",), -1,
                        (("time_h", 0.25),),
                        eligibility="waiting or headway must be recorded; total travel time rises with it"),
    "transfer_up": EditSpec("transfer_up", ("transfers", "interchanges"), +1.0, "transfers",
                            ("convenience",), ("financial",), -1,
                            eligibility="transfer count must be recorded; no compensating change"),
    "access_up": EditSpec("access_up", ("walking_h", "access_h"), +0.2, "hours",
                          ("convenience", "time"), ("financial",), -1, (("time_h", 0.2),),
                          eligibility="access or walking time must be recorded"),
}


def apply_edit(b: Bundle, spec: EditSpec, slot: int) -> Tuple[torch.Tensor, np.ndarray]:
    """Return edited attributes and the rows where the edit is eligible."""
    f = spec.feature_index(b)
    X = b.Xnum.clone()
    if f is None:
        return X, np.zeros(b.N, dtype=bool)
    eligible = (b.Xnum[:, slot, f] > 0).numpy()
    X[eligible, slot, f] = X[eligible, slot, f] + spec.delta
    names = b.meta["alt_feature_names"]
    for lname, ldelta in spec.linked:
        if lname in names:
            li = names.index(lname)
            X[eligible, slot, li] = X[eligible, slot, li] + ldelta
    return X, eligible


def edited_embeddings(b: Bundle, spec: EditSpec, slot: int, *, cache_dir=None):
    """Re-render the deterministic templates on the edited attributes and encode them.

    Only the template channel supports an edit exactly: re-generating LLM text for an edited
    attribute needs the generator, which is recorded as blocked when no generation budget
    is configured.
    """
    import dataclasses

    from omleu_experiments.sentences import build_source, template_sentences
    X, eligible = apply_edit(b, spec, slot)
    b_edit = dataclasses.replace(b, Xnum=X)
    E = build_source(b_edit, "template", cache_dir=cache_dir, tag=f"_edit_{spec.name}_s{slot}")
    return E, eligible
