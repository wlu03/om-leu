"""Coordinator-owned variant registry and run suites.

Every contrast in the report is a pair of names from this registry, so that the three
estimands stay separated:

* ensemble contribution   numeric_only            vs numeric_auxiliary
* semantic contribution   preserved_llm           vs preserved_template / _identity / _shuffled / _random
* structural contribution preserved_llm (or axis) vs plain_* with identical information
"""
from __future__ import annotations

from typing import Dict, List

from .runner import Variant

VARIANTS: Dict[str, Variant] = {}


def register(v: Variant) -> Variant:
    assert v.name not in VARIANTS, f"duplicate variant {v.name}"
    VARIANTS[v.name] = v
    return v


# --- numeric baselines -------------------------------------------------------------
register(Variant("numeric_only", reader=None, sentence_source="none", group="baseline",
                 notes="Stage 1 + Stage 2 with its own separately fitted temperature"))
register(Variant("numeric_stage1_only", reader=None, sentence_source="none", use_boost=False,
                 group="baseline", notes="Stage 1 alone, own temperature"))
register(Variant("numeric_auxiliary", reader="numeric_auxiliary", sentence_source="none", group="control",
                 notes="parameter-matched second numeric channel through the same mixture: ensemble contribution"))

# --- preserved Stage 3 on different sentence sources -------------------------------
for src, note in (("llm", "the cached frozen-LLM consequences"),
                  ("template", "deterministic factual templates from the generation allowlist"),
                  ("identity", "alternative-identity text only"),
                  ("shuffled", "a donor event's sentences, same alternative and axis"),
                  ("random", "seeded random unit vectors")):
    register(Variant(f"preserved_{src}", reader="preserved", sentence_source=src,
                     group="core" if src in ("llm", "template") else "control", notes=note))

# --- ordinary models with identical information ------------------------------------
register(Variant("plain_mean_llm", reader="plain", reader_kw={"inputs": ("sent",), "slots": False},
                 sentence_source="llm", group="control", notes="two-layer network on the mean sentence embedding"))
register(Variant("plain_slots_llm", reader="plain", reader_kw={"inputs": ("sent",), "slots": True},
                 sentence_source="llm", group="control", notes="axis slots preserved, so pooling is not the confound"))
register(Variant("plain_slots_z_llm", reader="plain", reader_kw={"inputs": ("sent", "z"), "slots": True},
                 sentence_source="llm", group="control", notes="slots plus person covariates"))
register(Variant("plain_slots_template", reader="plain", reader_kw={"inputs": ("sent",), "slots": True},
                 sentence_source="template", group="control", notes="ordinary model on templates"))
register(Variant("plain_all_inputs_llm", reader="plain", reader_kw={"inputs": ("sent", "z", "x"), "slots": True},
                 sentence_source="llm", group="control", notes="everything, no structure"))

SUITES: Dict[str, List[str]] = {
    "smoke": ["numeric_only", "preserved_llm"],
    "pilot": ["numeric_only", "preserved_llm", "preserved_template", "plain_slots_llm", "numeric_auxiliary"],
    "core": ["numeric_only", "numeric_stage1_only", "numeric_auxiliary", "preserved_llm", "preserved_template",
             "preserved_identity", "preserved_shuffled", "preserved_random", "plain_mean_llm", "plain_slots_llm",
             "plain_slots_z_llm", "plain_slots_template", "plain_all_inputs_llm"],
    "transfer": [],
    "optional": [],
}
