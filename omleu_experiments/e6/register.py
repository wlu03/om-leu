"""E6 variants: learning curves and the six ordered transfers."""
from omleu_experiments.readers import READERS
from omleu_experiments.registry import SUITES, register
from omleu_experiments.runner import Variant

from .transfer import TransferReader

READERS["transfer"] = TransferReader

_LC = []
for frac in (0.1, 0.25, 0.5, 1.0):
    for base, kw, src in (("preserved", {}, "llm"), ("axis", {"sensitivity": "global"}, "llm"),
                          ("axis", {"sensitivity": "global"}, "template")):
        name = f"lc_{base}_{src}_{int(frac*100):03d}"
        if name in [v.name for v in _LC]:
            continue
        _LC.append(Variant(name, reader=base, reader_kw=kw, sentence_source=src, train_fraction=frac,
                           group="transfer", notes=f"learning curve at {int(frac*100)}% of training respondents"))
_TR = []
DS = ("swissmetro", "optima", "lpmc")
for s in DS:
    for t in DS:
        if s == t:
            continue
        _TR.append(Variant(f"transfer_{s}_to_{t}", reader="transfer",
                           reader_kw={"source": s, "axis_kw": {"sensitivity": "global"}, "freeze": True},
                           sentence_source="llm", group="transfer",
                           notes=f"axis scorer fitted on {s}, frozen; valuation and calibration refitted on {t}"))
        _TR.append(Variant(f"transfer_{s}_to_{t}_finetune", reader="transfer",
                           reader_kw={"source": s, "axis_kw": {"sensitivity": "global"}, "freeze": False},
                           sentence_source="llm", group="transfer",
                           notes="full fine-tuning comparator"))
for v in _LC + _TR:
    register(v)
SUITES["e6_learning_curve"] = [v.name for v in _LC]
SUITES["e6_transfer"] = [v.name for v in _TR]
SUITES["transfer"] = [v.name for v in _LC + _TR]
