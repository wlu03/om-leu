"""Regression tests for pipeline errors found in the 2026-09 audit.

1. The outcomes cache key must depend on the rendered alternative
   attributes, not only on (customer, asin, seed, c_d): real-slate
   datasets vary price / travel time per event for the same asin.
2. Per-alternative history counts (``is_repeat`` / ``purchase_count``)
   must be strictly PRIOR to the event, for train events too (they used
   to count the customer's whole train history, including later events).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# 1. cache key includes the alternative attributes
# --------------------------------------------------------------------------- #


def test_cache_key_folds_in_alt_attributes() -> None:
    from src.outcomes.generate import build_cache_prompt_version

    base = dict(prompt_version="v6_travel_anchored", K=5, model_id="m", c_d="ctx")
    legacy = build_cache_prompt_version(**base)
    assert legacy == "v6_travel_anchored-K5-m-cd" + legacy.split("-cd")[1]
    a = build_cache_prompt_version(**base, alt={"title": "Car", "price": 65.0, "travel_time": "1 h 57 min"})
    b = build_cache_prompt_version(**base, alt={"title": "Car", "price": 84.0, "travel_time": "1 h 57 min"})
    same_as_a = build_cache_prompt_version(**base, alt={"travel_time": "1 h 57 min", "price": 65.0, "title": "Car"})
    assert a != b and a == same_as_a
    assert a.startswith(legacy + "-alt")


def test_generate_outcomes_caches_per_alternative_attributes(tmp_path: Path) -> None:
    from src.outcomes import generate as gen
    from src.outcomes.cache import OutcomesCache

    calls: list[dict] = []

    class _Client:
        model_id = "open/x"

        def generate(self, messages, **kw):
            calls.append(messages[1]["content"])
            return gen.GenerationResult(
                text="\n".join(f"outcome {len(calls)} line {i}" for i in range(5)),
                finish_reason="stop", model_id="open/x",
            )

    cache = OutcomesCache(tmp_path / "o.sqlite")
    common = dict(K=5, seed=0, prompt_version="v6_travel_anchored", client=_Client(), cache=cache)
    alt1 = {"title": "Car", "category": "commute", "price": 65.0, "popularity_rank": "top 50%",
            "travel_time": "1 h 57 min"}
    alt2 = dict(alt1, price=84.0, travel_time="2 h 10 min")
    p1 = gen.generate_outcomes("r1", "car", "ctx", alt1, **common)
    p2 = gen.generate_outcomes("r1", "car", "ctx", alt2, **common)
    p1_again = gen.generate_outcomes("r1", "car", "ctx", alt1, **common)
    cache.close()
    assert len(calls) == 2, "two different attribute sets -> two generations"
    assert p1.outcomes != p2.outcomes
    assert p1_again.outcomes == p1.outcomes  # cache hit for identical attributes


# --------------------------------------------------------------------------- #
# 2. prior-only history counts
# --------------------------------------------------------------------------- #


def _tiny_pipeline(tmp_path: Path):
    """Two customers, one repeats mode 'a' three times then picks 'b'."""
    from src.data.choice_sets import build_choice_sets

    base = pd.Timestamp("2020-01-01")
    rows = []
    # customer c1: a, a, b, a, b  (dates 0,1,2,3,4 days); split: train x3, val, test
    for i, (asin, split) in enumerate([("a", "train"), ("a", "train"), ("b", "train"),
                                        ("a", "val"), ("b", "test")]):
        rows.append(dict(customer_id="c1", order_date=base + pd.Timedelta(days=i), asin=asin,
                         category="cat", price=1.0 + i, title=f"item {asin}", split=split,
                         routine=0, novelty=1, recency_days=999.0, cat_affinity=0,
                         popularity=1, brand="", slate="a|b|c"))
    for i, (asin, split) in enumerate([("c", "train"), ("c", "train"), ("c", "train"),
                                        ("c", "val"), ("c", "test")]):
        rows.append(dict(customer_id="c2", order_date=base + pd.Timedelta(days=i), asin=asin,
                         category="cat", price=2.0, title=f"item {asin}", split=split,
                         routine=0, novelty=1, recency_days=999.0, cat_affinity=0,
                         popularity=1, brand="", slate="a|b|c"))
    events = pd.DataFrame(rows)
    persons = pd.DataFrame({
        "customer_id": ["c1", "c2"], "age_bucket": "35-44", "income_bucket": "50-100k",
        "household_size": 2, "has_kids": 0, "city_size": "large", "education": 3,
        "health_rating": 3, "risk_tolerance": 0, "purchase_frequency": 1.0, "novelty_rate": 0.5,
    })

    class _Adapter:
        name = "tiny"

        def suppress_fields_for_c_d(self):
            return ()

        def alt_text(self, row):
            return {"title": row.get("title", ""), "category": row.get("category", ""),
                    "price": float(row.get("price", 0.0) or 0.0), "popularity_rank": "top 50%",
                    "popularity_count": int(row.get("popularity", 0) or 0), "brand": ""}

    recs = build_choice_sets(events, persons, _Adapter(), seed=0, n_negatives=2,
                             real_slate_column="slate")
    return recs


def test_history_counts_are_strictly_prior_for_train_events(tmp_path: Path) -> None:
    recs = _tiny_pipeline(tmp_path)
    c1 = [r for r in recs if r["customer_id"] == "c1"]
    c1.sort(key=lambda r: r["order_date"])

    def count(rec, asin):
        return rec["alt_texts"][rec["choice_asins"].index(asin)]["purchase_count"]

    # event 0 (train, chose a): nothing before it
    assert count(c1[0], "a") == 0 and count(c1[0], "b") == 0
    # event 1 (train, chose a): one prior 'a'
    assert count(c1[1], "a") == 1 and c1[1]["alt_texts"][c1[1]["choice_asins"].index("a")]["is_repeat"] == 1.0
    # event 2 (train, chose b): two prior 'a', zero prior 'b' -- the old code
    # reported the customer's whole train history here (a=2 is fine, but b
    # would have been 1 - 1 = 0 only via the self-count hack, and 'a' at
    # event 0 would have been 2).
    assert count(c1[2], "a") == 2 and count(c1[2], "b") == 0
    # val event (chose a): all three train events are prior -> a=2, b=1
    assert count(c1[3], "a") == 2 and count(c1[3], "b") == 1
    # test event: val rows are NOT train history -> still a=2, b=1
    assert count(c1[4], "a") == 2 and count(c1[4], "b") == 1
    # never-chosen alternative 'c' for c1 is always 0
    assert all(count(r, "c") == 0 for r in c1)
