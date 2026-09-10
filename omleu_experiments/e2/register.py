"""E2 variants added to the coordinator's registry."""
from omleu_experiments.readers import READERS
from omleu_experiments.registry import SUITES, register
from omleu_experiments.runner import Variant

from .axis_reader import AxisReader

READERS["axis"] = AxisReader

_V = [
    Variant("axis_llm", reader="axis", reader_kw={"sensitivity": "global"}, sentence_source="llm",
            group="core", notes="axis-preserving reader on the cached consequences"),
    Variant("axis_template", reader="axis", reader_kw={"sensitivity": "global"}, sentence_source="template",
            group="core", notes="axis reader on equal-information templates"),
    Variant("axis_llm_anchored", reader="axis", reader_kw={"sensitivity": "global", "anchors": True},
            sentence_source="llm", group="core", notes="with anchor ordering supervision"),
    Variant("axis_llm_fixed_s", reader="axis", reader_kw={"sensitivity": "fixed"}, sentence_source="llm",
            group="ablation", notes="sensitivity fixed to one"),
    Variant("axis_llm_person_s", reader="axis", reader_kw={"sensitivity": "person"}, sentence_source="llm",
            group="ablation", notes="bounded person-specific sensitivity"),
    Variant("axis_llm_uniform_w", reader="axis", reader_kw={"sensitivity": "global", "w_hidden": 1},
            sentence_source="llm", group="ablation", notes="near-uniform valuation weights"),
    Variant("axis_llm_no_history", reader="axis", reader_kw={"sensitivity": "global", "use_history": False},
            sentence_source="llm", group="ablation", notes="covariates only, no admissible history"),
    Variant("axis_llm_shared_head", reader="axis", reader_kw={"sensitivity": "global", "share_head": True},
            sentence_source="llm", group="ablation", notes="one readout shared by all axes"),
    Variant("axis_shuffled", reader="axis", reader_kw={"sensitivity": "global"}, sentence_source="shuffled",
            group="control", notes="axis reader on donor sentences"),
    Variant("axis_identity", reader="axis", reader_kw={"sensitivity": "global"}, sentence_source="identity",
            group="control", notes="axis reader on alternative-identity text"),
]
for v in _V:
    register(v)
SUITES["e2"] = [v.name for v in _V]
SUITES["core"] = SUITES["core"] + ["axis_llm", "axis_template", "axis_llm_anchored"]
