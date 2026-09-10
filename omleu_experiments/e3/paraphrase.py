"""Paraphrases that preserve claims, quantities, negation, uncertainty and evidence status.

The templates are deterministic rewordings of the template channel, so a paraphrase never
introduces or removes a number, a negation or an evidence qualifier.  Paraphrase families
are held out for evaluation: a family seen in fitting is never scored.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import torch

from experiments.harness.data import Bundle

FAMILIES: Dict[str, List[Tuple[str, str]]] = {
    "recorded_to_measured": [(r"^Recorded ", "Measured ")],
    "colon_to_clause": [(r": ([^.]*)\.$", r" is \1.")],
    "passive": [(r"^Recorded (.*?) for (.*?): (.*)\.$", r"For \2, the recorded \1 is \3.")],
    "notrecorded": [(r"not recorded", "not established by the available evidence"),
                    (r"^No (.*?) is recorded for (.*)\.$", r"The available evidence does not establish \1 for \2.")],
}


def paraphrase_text(text: str, family: str) -> str:
    out = text
    for pat, rep in FAMILIES[family]:
        out = re.sub(pat, rep, out)
    return out


def paraphrase_embeddings(b: Bundle, family: str, *, cache_dir=None) -> torch.Tensor:
    """Template sentences rewritten by one family, then encoded."""
    from omleu_experiments.sentences import build_source, template_sentences

    def builder(bb: Bundle, i: int, j: int):
        return {k: paraphrase_text(v, family) for k, v in template_sentences(bb, i, j).items()}

    return build_source(b, "custom", cache_dir=cache_dir, builder=builder, tag=f"_para_{family}")


def quantities(text: str) -> List[str]:
    return re.findall(r"-?\d+\.?\d*", text)


NEGATION = ("not ", "no ", "does not", "cannot")
UNCERTAINTY = ("not recorded", "not established", "unknown", "does not establish")


def preserves_claims(a: str, bb: str) -> bool:
    """A paraphrase must keep every quantity, the negation polarity and the evidence status.

    Polarity is checked as a property of the sentence rather than by matching individual
    words, so "No X is recorded" and "The available evidence does not establish X" count as
    the same claim while "X is recorded" does not.
    """
    if quantities(a) != quantities(bb):
        return False
    neg = lambda t: any(m in t.lower() for m in NEGATION)
    unc = lambda t: any(m in t.lower() for m in UNCERTAINTY)
    return neg(a) == neg(bb) and unc(a) == unc(bb)
