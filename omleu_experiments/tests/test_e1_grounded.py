"""E1 acceptance tests: allowlist, validators, and the zero-paid-request default."""
from __future__ import annotations

import json
import os

import pytest

from omleu_experiments.e1.generation import budget_configured, plan
from omleu_experiments.e1.prompts import GROUNDED_ALLOWLIST, PERSON_FREE_ALLOWLIST, grounded_messages
from omleu_experiments.e1.validators import numeric_support, unsupported_claim_scan, validate_consequence


def test_allowlist_excludes_label_derived_person_aggregates():
    for bad in ("novelty_rate", "purchase_frequency", "chosen_asin", "y"):
        assert bad not in GROUNDED_ALLOWLIST
    assert set(PERSON_FREE_ALLOWLIST) < set(GROUNDED_ALLOWLIST)


def test_prompt_forbids_invented_precision_and_preference_inference():
    msgs = grounded_messages({"cost": 1.18, "currency": "CHF"}, "car", K=5)
    sys = msgs[0]["content"].lower()
    for phrase in ("never invent", "parking", "delay probabilities", "never judge", "never infer a preference"):
        assert phrase in sys


def test_numeric_support_accepts_unit_conversions_and_flags_invention():
    supplied = {"time_h": 0.5, "cost_chf": 1.18}
    assert numeric_support("I travel for 30 minutes.", supplied)["unsupported"] == []
    assert numeric_support("I pay CHF 1.18.", supplied)["unsupported"] == []
    assert numeric_support("I wait 7.3 minutes for parking.", supplied)["unsupported"] == [7.3]


def test_claim_scan_catches_the_forbidden_claim_types():
    assert "affordability judgement" in unsupported_claim_scan("I spend a negligible CHF 1.18.")
    assert "parking availability" in unsupported_claim_scan("I can park easily near the shops.")
    assert "stated preference rather than outcome" in unsupported_claim_scan("I enjoy a comfortable ride.")
    assert unsupported_claim_scan("Recorded travel time: 12 minutes.") == []


def test_validator_requires_assumptions_for_derived_and_missing_for_unknown():
    supplied = {"cost_chf": 1.18}
    ok = validate_consequence({"axis": "financial", "text": "I pay CHF 1.18.", "evidence_status": "observed",
                               "supporting_fields": ["cost"], "assumptions": [], "missing_information": []},
                              GROUNDED_ALLOWLIST, supplied)
    assert ok["ok"], ok["errors"]
    bad = validate_consequence({"axis": "financial", "text": "I pay about CHF 2 per kilometre.",
                                "evidence_status": "derived", "supporting_fields": ["cost"],
                                "assumptions": [], "missing_information": []}, GROUNDED_ALLOWLIST, supplied)
    assert not bad["ok"] and any("assumption" in e for e in bad["errors"])
    bad2 = validate_consequence({"axis": "financial", "text": "Parking is free.", "evidence_status": "unknown",
                                 "supporting_fields": [], "assumptions": [], "missing_information": []},
                                GROUNDED_ALLOWLIST, supplied)
    assert not bad2["ok"]


def test_generation_defaults_to_zero_paid_requests(monkeypatch):
    monkeypatch.delenv("OMLEU_GENERATION_BUDGET", raising=False)
    assert budget_configured() is None
    p = plan("optima", 7, "grounded", 1906, 3, "test-rev")
    assert p.status == "blocked_generation_budget" and p.requests == 5718 and p.sentences == 28590
    monkeypatch.setenv("OMLEU_GENERATION_BUDGET", json.dumps({"requests": 10, "provider": "ollama"}))
    p2 = plan("optima", 7, "grounded", 1906, 3, "test-rev")
    assert p2.status == "blocked_generation_budget" and "budget allows 10" in p2.reason
