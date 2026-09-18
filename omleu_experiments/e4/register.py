"""E4 variants."""
from omleu_experiments.readers import READERS
from omleu_experiments.registry import SUITES, register
from omleu_experiments.runner import Variant

from .complementarity import MixtureAwareReader

READERS["mixture_aware"] = MixtureAwareReader

_V = [
    Variant("mixture_aware_preserved", reader="mixture_aware", reader_kw={"base": "preserved"},
            sentence_source="llm", group="core",
            notes="preserved reader, pretrained independently then fine-tuned on the mixture loss"),
    Variant("mixture_aware_plain", reader="mixture_aware",
            reader_kw={"base": "plain", "base_kw": {"inputs": ("sent",), "slots": True}},
            sentence_source="llm", group="core",
            notes="ordinary reader with the same training objective: isolates objective from architecture"),
    Variant("mixture_aware_template", reader="mixture_aware", reader_kw={"base": "preserved"},
            sentence_source="template", group="control", notes="mixture-aware training on templates"),
]
for v in _V:
    register(v)
SUITES["e4"] = [v.name for v in _V]
