"""The frozen experiment matrix and its declared contrasts.

Importing this module registers every experiment's variants.  The matrix is finite and
chosen in advance; it is not every combination of the available options.  Contrasts are
declared here so that the report cannot be assembled by searching for a favourable pair.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from . import registry
from .registry import SUITES, VARIANTS
from .runner import Variant

import omleu_experiments.e1.register  # noqa: F401,E402
import omleu_experiments.e2.register  # noqa: F401,E402
import omleu_experiments.e4.register  # noqa: F401,E402
import omleu_experiments.e6.register  # noqa: F401,E402
import omleu_experiments.e7.register  # noqa: F401,E402

# combined model: grounded-allowlist templates read by the axis model, trained mixture-aware
registry.register(Variant("combined_axis_template_mixture_aware", reader="mixture_aware",
                          reader_kw={"base": "axis", "base_kw": {"sensitivity": "global"}},
                          sentence_source="template", group="core",
                          notes="axis reader on templates, pretrained then fine-tuned on the mixture loss"))
registry.register(Variant("combined_axis_llm_mixture_aware", reader="mixture_aware",
                          reader_kw={"base": "axis", "base_kw": {"sensitivity": "global"}},
                          sentence_source="llm", group="core",
                          notes="axis reader on the cached consequences, mixture-aware training"))

#: (name, baseline, estimand, what the contrast is evidence about)
CONTRASTS: List[Tuple[str, str, str, str]] = [
    ("numeric_auxiliary", "numeric_only", "ensemble",
     "a second, non-semantic predictor mixed through the same calibration"),
    ("preserved_llm", "numeric_only", "ensemble+semantic",
     "the shipped full model against its own numeric channel"),
    ("preserved_llm", "numeric_auxiliary", "semantic",
     "consequences against a matched non-semantic channel with the same mixture"),
    ("preserved_llm", "preserved_shuffled", "semantic",
     "real consequences against donor consequences for the same alternative and axis"),
    ("preserved_llm", "preserved_identity", "semantic",
     "real consequences against alternative-identity text"),
    ("preserved_llm", "preserved_random", "semantic",
     "real consequences against random unit vectors"),
    ("preserved_llm", "preserved_template", "semantic",
     "generated consequences against equal-information deterministic templates"),
    ("preserved_llm", "plain_slots_llm", "structural",
     "the shipped reader against an ordinary network with identical information"),
    ("axis_llm", "plain_slots_llm", "structural",
     "the axis-preserving reader against an ordinary network with identical information"),
    ("axis_llm", "preserved_llm", "structural",
     "axis-preserving reader against the shipped reader"),
    ("axis_template", "axis_llm", "semantic",
     "templates against generated consequences inside the axis reader"),
    ("mixture_aware_preserved", "preserved_llm", "structural",
     "mixture-aware training against independent training, same architecture"),
    ("mixture_aware_plain", "plain_slots_llm", "structural",
     "the same objective change for an ordinary reader"),
    ("gate_conditional_llm", "gate_global_llm", "structural",
     "a conditional gate against a global mixture weight"),
    ("gate_conditional_llm", "gate_conditional_numeric_aux", "semantic",
     "the same gate over a matched non-semantic channel: gate capacity control"),
]

#: the confirmatory family: these are prespecified and multiplicity-corrected in the report
CONFIRMATORY = ["preserved_llm vs numeric_auxiliary", "preserved_llm vs preserved_shuffled",
                "preserved_llm vs preserved_template", "preserved_llm vs plain_slots_llm",
                "axis_llm vs plain_slots_llm"]

SUITES["core_matrix"] = [
    "numeric_only", "numeric_stage1_only", "numeric_auxiliary",
    "preserved_llm", "preserved_template", "preserved_identity", "preserved_shuffled", "preserved_random",
    "plain_mean_llm", "plain_slots_llm", "plain_slots_z_llm", "plain_slots_template", "plain_all_inputs_llm",
    "axis_llm", "axis_template", "axis_llm_anchored",
]
SUITES["core_matrix_extended"] = SUITES["core_matrix"] + [
    "axis_llm_fixed_s", "axis_llm_person_s", "axis_llm_uniform_w", "axis_llm_no_history", "axis_llm_shared_head",
    "axis_shuffled", "axis_identity",
    "mixture_aware_preserved", "mixture_aware_plain", "mixture_aware_template",
    "combined_axis_template_mixture_aware", "combined_axis_llm_mixture_aware",
]
