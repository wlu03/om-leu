"""Programmatic validation of generated consequences.

Checks that can be made mechanically: schema and axis coverage, evidence-status vocabulary,
supporting fields drawn from the allowlist, and every numeral in the text matching a
supplied quantity under a declared unit conversion.  Everything else - whether a sentence is
a fair reading of the evidence - is *unverified* and is left to the human audit; the
generator's own judgement never certifies it.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from omleu_experiments.contracts import AXES, EVIDENCE_STATUS

NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
# a numeral in the text is supported if it matches a supplied quantity in one of these units
CONVERSIONS = {"as_is": 1.0, "hours_to_minutes": 60.0, "hundreds": 100.0, "km_to_m": 1000.0}


def _numbers(text: str) -> List[float]:
    return [float(m.group().replace(",", ".")) for m in NUM.finditer(text)]


def numeric_support(text: str, supplied: Mapping[str, float], tol: float = 0.051) -> Dict[str, object]:
    """Which numerals in a sentence match a supplied quantity, and which do not."""
    vals = []
    for v in supplied.values():
        if v is None:
            continue
        for f in CONVERSIONS.values():
            vals.append(float(v) * f)
    vals += [0.0, 1.0, 2.0]                     # counts and trivial integers are always admissible
    unsupported = []
    for n in _numbers(text):
        if not any(abs(n - v) <= max(tol, 0.02 * abs(v)) for v in vals):
            unsupported.append(n)
    return {"n_numerals": len(_numbers(text)), "unsupported": unsupported,
            "supported": len(_numbers(text)) - len(unsupported)}


def validate_consequence(obj: Mapping[str, object], allowlist: Sequence[str],
                         supplied: Mapping[str, float]) -> Dict[str, object]:
    errors: List[str] = []
    if obj.get("axis") not in AXES:
        errors.append(f"axis {obj.get('axis')!r} is not one of the five axes")
    if obj.get("evidence_status") not in EVIDENCE_STATUS:
        errors.append(f"evidence_status {obj.get('evidence_status')!r} is not in the vocabulary")
    text = str(obj.get("text", ""))
    if not text or len(text) > 400:
        errors.append("text is empty or too long")
    bad_fields = [f for f in obj.get("supporting_fields", []) if f not in allowlist]
    if bad_fields:
        errors.append(f"supporting_fields outside the allowlist: {bad_fields}")
    ns = numeric_support(text, supplied)
    if obj.get("evidence_status") == "observed" and ns["unsupported"]:
        errors.append(f"observed sentence states unsupplied quantities {ns['unsupported']}")
    if obj.get("evidence_status") == "derived" and not obj.get("assumptions"):
        errors.append("derived sentence states no assumption")
    if obj.get("evidence_status") == "unknown" and not obj.get("missing_information"):
        errors.append("unknown sentence names no missing quantity")
    return {"ok": not errors, "errors": errors, "numeric": ns}


JUDGEMENT_PATTERNS = [
    (re.compile(r"\b(negligible|cheap|affordable|expensive|a bargain|reasonably priced)\b", re.I), "affordability judgement"),
    (re.compile(r"\b(park(ing)? (is|will be) (easy|available|free)|easily park|park easily)\b", re.I), "parking availability"),
    (re.compile(r"\b(rarely delayed|always on time|never late|guaranteed)\b", re.I), "delay claim"),
    (re.compile(r"\b(as a (retiree|student|man|woman)|at my age|given my income)[, ].{0,40}\b(prefer|enjoy|value|like)\b", re.I),
     "demographic preference inference"),
    (re.compile(r"\b(I (enjoy|love|prefer|value|appreciate))\b", re.I), "stated preference rather than outcome"),
]


def unsupported_claim_scan(text: str) -> List[str]:
    """Pattern scan for the claim types the grounded prompt forbids.  A pattern scan measures
    what it matches and nothing more; it is a lower bound on unsupported claims."""
    return [label for pat, label in JUDGEMENT_PATTERNS if pat.search(text)]


def numeric_support_report(bundle, texts_by_row_alt, supplied_by_row_alt, limit: Optional[int] = None) -> Dict:
    """Aggregate numeric support and forbidden-claim rates over cached generations."""
    n_sent = n_unsupported_sent = n_num = n_unsupported_num = 0
    claims: Dict[str, int] = {}
    examples: List[Dict] = []
    for key, sentences in list(texts_by_row_alt.items())[:limit]:
        supplied = supplied_by_row_alt[key]
        for k, s in enumerate(sentences):
            n_sent += 1
            ns = numeric_support(s, supplied)
            n_num += ns["n_numerals"]; n_unsupported_num += len(ns["unsupported"])
            hits = unsupported_claim_scan(s)
            for h in hits:
                claims[h] = claims.get(h, 0) + 1
            if ns["unsupported"] or hits:
                n_unsupported_sent += 1
                if len(examples) < 25:
                    examples.append({"row": key[0], "alt": key[1], "axis": k, "text": s,
                                     "unsupported_numerals": ns["unsupported"], "claims": hits})
    return {"sentences": n_sent, "sentences_with_a_flag": n_unsupported_sent,
            "flagged_rate": n_unsupported_sent / max(n_sent, 1),
            "numerals": n_num, "unsupported_numerals": n_unsupported_num,
            "unsupported_numeral_rate": n_unsupported_num / max(n_num, 1),
            "claim_counts": claims, "examples": examples,
            "note": "a pattern scan is a lower bound; a sentence with no flag is not certified as grounded"}
