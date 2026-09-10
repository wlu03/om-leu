"""Typed, versioned records shared by every experiment.

Contract version is part of every artifact's provenance; bump ``CONTRACT_VERSION``
when a field's meaning changes.  Labels live in their own field and are never part of
a generation input.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CONTRACT_VERSION = "1.0.0"

# Ordering semantics.  "calendar" is real wall-clock time; "task_index" is the
# presentation order of a stated-preference questionnaire (NOT time); "synthetic" is a
# date manufactured by a preparation script; "none" means no ordering field exists.
ORDER_KINDS = ("calendar", "task_index", "synthetic", "none")
AXES = ("financial", "time", "comfort", "convenience", "reliability")
EVIDENCE_STATUS = ("observed", "derived", "unknown")


def stable_hash(obj: Any) -> str:
    """SHA-256 of a canonical JSON rendering; used for cache and provenance identity."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class Event:
    """One choice occasion.  ``chosen_index`` is supervised information: it must never
    be copied into a :class:`GenerationInput` or into any feature builder."""

    dataset: str
    event_id: str
    person_id: str
    alt_ids: Tuple[str, ...]              # canonical alternative name per slot
    avail: Tuple[bool, ...]               # availability mask per slot
    attrs: Mapping[str, Tuple[float, ...]]        # attribute name -> value per slot
    covariates: Mapping[str, float]               # person covariates z_i
    history: Mapping[str, Tuple[float, ...]]      # admissible history per slot
    chosen_index: int                             # supervised field, kept separate
    household_id: Optional[str] = None
    order_value: Optional[float] = None
    order_kind: str = "none"

    def __post_init__(self):
        J = len(self.alt_ids)
        assert len(self.avail) == J, "avail must have one entry per slot"
        assert self.order_kind in ORDER_KINDS, f"bad order_kind {self.order_kind}"
        assert 0 <= self.chosen_index < J, "chosen_index out of range"
        assert self.avail[self.chosen_index], "chosen alternative must be available"
        for k, v in self.attrs.items():
            assert len(v) == J, f"attribute {k} has {len(v)} values, expected {J}"

    @property
    def cluster_id(self) -> str:
        """The independent sampling unit for the bootstrap: household when known."""
        return self.household_id or self.person_id


@dataclass(frozen=True)
class GenerationInput:
    """Everything a generator may see.  Constructed only through :func:`generation_input`,
    which enforces the allowlist, so a label can never reach a prompt."""

    dataset: str
    event_id: str
    person_id: str
    alt_id: str
    axis: str
    fields: Mapping[str, Any]
    allowlist: Tuple[str, ...]
    history_policy: str                    # e.g. "strict_prefix_own_events"

    @property
    def input_hash(self) -> str:
        return stable_hash({"f": dict(self.fields), "a": self.alt_id, "x": self.axis,
                            "p": self.history_policy, "w": list(self.allowlist)})


FORBIDDEN_GENERATION_KEYS = (
    "chosen_index", "chosen_asin", "y", "label", "choice", "target",
    "novelty_rate", "purchase_frequency",   # person aggregates over the person's own labels
)


def generation_input(event: Event, slot: int, axis: str, fields: Mapping[str, Any],
                     allowlist: Sequence[str], history_policy: str) -> GenerationInput:
    """Build a generation input, rejecting label-derived or non-allowlisted fields."""
    bad = [k for k in fields if k not in allowlist]
    if bad:
        raise ValueError(f"fields outside the allowlist: {bad}")
    bad = [k for k in fields if k.lower() in FORBIDDEN_GENERATION_KEYS]
    if bad:
        raise ValueError(f"label-derived fields in a generation input: {bad}")
    return GenerationInput(dataset=event.dataset, event_id=event.event_id, person_id=event.person_id,
                           alt_id=event.alt_ids[slot], axis=axis, fields=dict(fields),
                           allowlist=tuple(allowlist), history_policy=history_policy)


@dataclass(frozen=True)
class Consequence:
    """One generated (or templated) consequence sentence with its provenance."""

    dataset: str
    event_id: str
    alt_id: str
    axis: str
    text: str
    evidence_status: str                  # observed | derived | unknown
    supporting_fields: Tuple[str, ...]
    assumptions: Tuple[str, ...]
    missing_information: Tuple[str, ...]
    prompt_version: str
    generator_revision: str
    decoding: Mapping[str, Any]
    input_hash: str
    encoder_revision: str = ""
    embedding_hash: str = ""

    def __post_init__(self):
        assert self.evidence_status in EVIDENCE_STATUS, self.evidence_status

    @property
    def cache_identity(self) -> str:
        """Cache identity: content, alternative, axis, history policy (via input_hash),
        prompt revision, generator revision and decoding settings."""
        return stable_hash({"i": self.input_hash, "p": self.prompt_version,
                            "g": self.generator_revision, "d": dict(self.decoding), "x": self.axis})


@dataclass
class OutcomeModelOutput:
    """Factors of a semantic channel, kept for diagnostics (E2/E3/E5)."""

    scores: Any            # (J, K) per-alternative per-axis outcome scores
    weights: Any           # (K,) person weights over axes
    sensitivity: float     # s_i >= 0
    utilities: Any         # (J,)
    avail: Any             # (J,) bool
    probs: Any             # (J,)


@dataclass
class Prediction:
    """Per-event prediction record written to disk for every evaluated event."""

    dataset: str
    event_id: str
    person_id: str
    cluster_id: str
    split: str
    fold: Optional[int]
    master_seed: int
    variant: str
    protocol: str
    alt_ids: List[str]
    avail: List[bool]
    numeric_logits: List[float]
    semantic_logprobs: Optional[List[float]]
    final_logprobs: List[float]
    chosen_index: int
    calibration: Dict[str, float]
    lineage: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


STATUSES = ("completed", "failed", "blocked_data", "blocked_generation_budget",
            "blocked_annotations", "blocked_resources", "not_run")
