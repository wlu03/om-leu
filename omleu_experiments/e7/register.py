"""E7 variants: a small conditional mixture gate and its controls.

The gate is secondary: it is only meaningful once the global-mixture baselines work, and a
gain from it may be gate capacity rather than anything about consequences, which is why the
same gate is fitted over a matched non-semantic channel.
"""
from omleu_experiments.registry import SUITES, register
from omleu_experiments.runner import Variant

_V = [
    Variant("gate_global_llm", reader="preserved", sentence_source="llm", calibrator="global",
            group="optional", notes="global scalar mixture weight (reference for the gate)"),
    Variant("gate_conditional_llm", reader="preserved", sentence_source="llm", calibrator="gate",
            gate_l2=1.0, group="optional", notes="conditional gate on history length, missingness, set size"),
    Variant("gate_conditional_llm_weak_shrinkage", reader="preserved", sentence_source="llm",
            calibrator="gate", gate_l2=0.1, group="optional", notes="less shrinkage towards the global gate"),
    Variant("gate_conditional_numeric_aux", reader="numeric_auxiliary", sentence_source="none",
            calibrator="gate", gate_l2=1.0, group="optional",
            notes="same gate over a matched non-semantic channel: gate capacity control"),
    Variant("gate_conditional_template", reader="preserved", sentence_source="template", calibrator="gate",
            gate_l2=1.0, group="optional", notes="same gate over the template channel"),
]
for v in _V:
    register(v)
SUITES["e7"] = [v.name for v in _V]
SUITES["optional"] = SUITES.get("optional", []) + [v.name for v in _V]
