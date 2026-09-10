"""Resumable generation adapter and dry-run manifest.

Default is zero new paid requests.  With no bounded budget configured the adapter builds the
manifest, counts requests, reports what a run would cost in requests and tokens, and records
the experiment as ``blocked_generation_budget``.  Templates and cached variants still run, and
a template variant is never published under the name of a generated one.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from omleu_experiments.contracts import AXES, stable_hash

from .prompts import GENERATION_SPEC, GROUNDED_ALLOWLIST, PERSON_FREE_ALLOWLIST, grounded_messages


@dataclass
class GenerationPlan:
    dataset: str
    seed: int
    variant: str
    n_events: int
    n_alternatives: int
    requests: int
    sentences: int
    allowlist: Sequence[str]
    prompt_version: str
    generator_revision: str
    decoding: Dict
    status: str
    reason: str

    def to_json(self) -> Dict:
        d = self.__dict__.copy(); d["allowlist"] = list(self.allowlist); return d


def budget_configured() -> Optional[Dict]:
    """A bounded generation budget must be declared explicitly, for example
    ``OMLEU_GENERATION_BUDGET='{"requests": 2000, "provider": "ollama", "model": "qwen3-vl:8b-instruct"}'``."""
    raw = os.environ.get("OMLEU_GENERATION_BUDGET")
    if not raw:
        return None
    try:
        b = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return b if int(b.get("requests", 0)) > 0 else None


def plan(dataset: str, seed: int, variant: str, n_events: int, n_alternatives: int,
         generator_revision: str) -> GenerationPlan:
    budget = budget_configured()
    requests = n_events * n_alternatives
    allow = PERSON_FREE_ALLOWLIST if variant.endswith("_no_person") else GROUNDED_ALLOWLIST
    status, reason = ("not_run", "budget configured; run with the generation command")
    if budget is None:
        status = "blocked_generation_budget"
        reason = ("no bounded generation budget is configured; set OMLEU_GENERATION_BUDGET to run. "
                  f"A full pass needs {requests} requests of {len(AXES)} sentences each.")
    elif requests > int(budget.get("requests", 0)):
        status = "blocked_generation_budget"
        reason = f"plan needs {requests} requests, budget allows {budget.get('requests')}"
    return GenerationPlan(dataset=dataset, seed=seed, variant=variant, n_events=n_events,
                          n_alternatives=n_alternatives, requests=requests, sentences=requests * len(AXES),
                          allowlist=allow, prompt_version=f"e1_{variant}",
                          generator_revision=generator_revision, decoding=GENERATION_SPEC["decoding"],
                          status=status, reason=reason)


def write_manifests(artifact_root: Path, datasets: Sequence[str], seeds: Sequence[int],
                    sizes: Dict[str, Dict[str, int]], generator_revision: str) -> List[Dict]:
    out = []
    for ds in datasets:
        for seed in seeds:
            for variant in ("grounded", "grounded_no_person", "existing_prompt_replication"):
                p = plan(ds, seed, variant, sizes[ds]["events"], sizes[ds]["alternatives"], generator_revision)
                out.append(p.to_json())
    d = Path(artifact_root) / "e1_generation"
    d.mkdir(parents=True, exist_ok=True)
    (d / "generation_manifest.json").write_text(json.dumps(out, indent=1))
    return out
