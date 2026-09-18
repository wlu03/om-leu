"""E1 variants.  Only the template and cached-generation variants can run without a
generation budget; the grounded variants are registered so that their status is recorded."""
from omleu_experiments.registry import SUITES, register
from omleu_experiments.runner import Variant

_V = [
    Variant("e1_existing_prompt_preserved", reader="preserved", sentence_source="llm", group="core",
            notes="the cached consequences from the existing prompt, preserved reader"),
    Variant("e1_template_preserved", reader="preserved", sentence_source="template", group="core",
            notes="matched deterministic templates from exactly the generation allowlist"),
    Variant("e1_template_axis", reader="axis", reader_kw={"sensitivity": "global"}, sentence_source="template",
            group="core", notes="templates read by the axis model"),
]
for v in _V:
    register(v)
SUITES["e1"] = [v.name for v in _V]
