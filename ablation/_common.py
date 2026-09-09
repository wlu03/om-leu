"""Shared machinery for the ablation program in ablation/<name>/.

Every ablation folder holds ``model.py`` exposing

    NAME    folder name
    LABEL   one-line description used in the tables
    GROUP   "baseline" (how much comes from the LLM sentences vs from the designed model)
            or "design" (knock-outs inside the sentence model)
    CONFIG  keyword overrides for ``experiments.models.omleu2.Config``

The runner (``ablation/run.py``) builds OM-LEU 2 with those overrides under two protocols
(chronological within-person split, person-level split) and two mixture-weight estimators
(validation-fitted pi, out-of-fold pi), and stores one JSON per run in
``ablation/<name>/results/``.  ``ablation/make_tables.py`` turns them into ``results.md``
per folder and the summary in ``ablation/README.md``.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.models.omleu2 import Config, _build  # noqa: E402

ABL_DIR = Path(__file__).resolve().parent
DATASETS = ["swissmetro", "optima", "lpmc"]
SEEDS = [7, 11, 13]
REFERENCE = "full_model"

FULL = dict(members=5, temp=True)
PROTOCOLS = {
    "chrono": dict(),                                                    # chronological within-person split (paper tables)
    "person": dict(cold_start=True, struct=(("person", False),)),       # person-level re-split, person effects off
}
PI_FITS = {"val": dict(pi_fit="val"), "oof": dict(pi_fit="oof")}


def discover() -> Dict[str, object]:
    """name -> imported model module, for every ablation/<name>/model.py."""
    out = {}
    for d in sorted(ABL_DIR.iterdir()):
        f = d / "model.py"
        if d.is_dir() and f.exists() and not d.name.startswith("_"):
            spec = importlib.util.spec_from_file_location(f"ablation.{d.name}.model", f)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            out[d.name] = mod
    return out


def build(mod, protocol: str, pi: str):
    kw = {**FULL, **PROTOCOLS[protocol], **PI_FITS[pi], **mod.CONFIG}
    cfg = Config(**kw)
    return lambda b, seed: _build(b, seed, cfg)


def result_path(name: str, protocol: str, pi: str, dataset: str, seed: int) -> Path:
    return ABL_DIR / name / "results" / f"{protocol}_pi{pi}_{dataset}_seed{seed}.json"


def load_results(name: str) -> List[dict]:
    out = []
    for f in sorted((ABL_DIR / name / "results").glob("*.json")) if (ABL_DIR / name / "results").exists() else []:
        d = json.loads(f.read_text())
        if "error" in d:
            continue
        out.append(d)
    return out


def paired_delta(a: List[float], b: List[float], draws: int = 2000, seed: int = 0):
    """mean(a - b) with a bootstrap CI over events; a, b per-event NLL of the same events."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or len(a) == 0:
        return None
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (draws, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def fmt_delta(t) -> str:
    if t is None:
        return "—"
    m, lo, hi = t
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{m:+.4f}{star} [{lo:+.4f}, {hi:+.4f}]"
