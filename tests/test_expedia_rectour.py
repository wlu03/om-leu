"""Tests for the Expedia RecTour onboarding + open-weight provider path.

Hermetic: the fixture generator writes a RecTour-shaped raw file, the
prepare script turns it into the four CSVs, and the choice-set builder
consumes real displayed slates. No LLM, no network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixture: raw -> prepared, once per session
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def prepared_dir(tmp_path_factory) -> Path:
    from scripts.make_expedia_rectour_fixture import make_fixture

    raw_dir = tmp_path_factory.mktemp("rectour_raw")
    raw, amen = make_fixture(raw_dir, n_users=40, n_props=600, n_dest=8, seed=3)
    out = tmp_path_factory.mktemp("rectour_prepared")
    rc = subprocess.call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_expedia_rectour.py"),
        "--raw", str(raw), "--amenities", str(amen), "--out", str(out),
        "--J", "6", "--min-events", "3", "--max-users", "0", "--seed", "1",
    ])
    assert rc == 0
    return out


def test_prepare_writes_all_four_files_and_summary(prepared_dir: Path) -> None:
    for name in ("events.csv", "persons.csv", "impressions.csv", "properties.csv",
                 "prepare_summary.json"):
        assert (prepared_dir / name).exists(), name
    summary = json.loads((prepared_dir / "prepare_summary.json").read_text())
    assert summary["J"] == 6
    assert summary["n_events"] > 0
    assert set(summary["label_kind_counts"]) <= {"book", "click"}


def test_prepare_slates_have_exactly_J_unique_members_including_chosen(prepared_dir: Path) -> None:
    ev = pd.read_csv(prepared_dir / "events.csv", dtype={"prop_id": str, "slate_prop_ids": str})
    for prop, slate in zip(ev["prop_id"], ev["slate_prop_ids"]):
        members = slate.split("|")
        assert len(members) == 6
        assert len(set(members)) == 6
        assert prop in members
    # Booked property wins over clicks when both exist.
    imp = pd.read_csv(prepared_dir / "impressions.csv", dtype={"search_id": str, "prop_id": str})
    assert set(imp.columns) >= {"search_id", "prop_id", "rank", "price_bucket",
                                "star_rating", "review_rating", "review_count",
                                "is_free_cancellation", "is_drr", "is_travel_ad"}
    # Every slate member of every event has an impressions row.
    keys = set(zip(imp["search_id"], imp["prop_id"]))
    for sid, slate in zip(ev["search_id"].astype(str), ev["slate_prop_ids"]):
        for m in slate.split("|"):
            assert (sid, m) in keys


def test_prepare_persons_use_only_early_history(prepared_dir: Path) -> None:
    ev = pd.read_csv(prepared_dir / "events.csv")
    persons = pd.read_csv(prepared_dir / "persons.csv", dtype={"user_id": str})
    counts = ev.groupby("user_id").size()
    hist = persons.set_index("user_id")["n_searches_hist"]
    for uid, n in counts.items():
        assert hist[str(uid)] == int(np.ceil(0.8 * n))
    assert set(persons["party_size_bucket"]) <= {"solo", "pair", "three", "four", "five_plus"}
    assert set(persons["has_kids"]) <= {0, 1}


def test_prepare_book_only_mode_keeps_only_transactions(tmp_path: Path) -> None:
    from scripts.make_expedia_rectour_fixture import make_fixture

    raw, _ = make_fixture(tmp_path / "raw", n_users=30, n_props=400, n_dest=6, seed=5)
    out = tmp_path / "prep"
    rc = subprocess.call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_expedia_rectour.py"),
        "--raw", str(raw), "--out", str(out), "--J", "5", "--min-events", "1",
        "--max-users", "0", "--label-mode", "book",
    ])
    assert rc == 0
    ev = pd.read_csv(out / "events.csv")
    assert set(ev["label_kind"]) == {"book"}


def test_prepare_accepts_flattened_layout(tmp_path: Path) -> None:
    """One row per impression (the layout the RecTour-Ranking repo used)."""
    from scripts.make_expedia_rectour_fixture import make_fixture
    from scripts.prepare_expedia_rectour import IMPR_FIELDS

    raw, _ = make_fixture(tmp_path / "raw", n_users=20, n_props=300, n_dest=5, seed=9)
    nested = pd.read_csv(raw, dtype=str)
    rows = []
    for rec in nested.to_dict("records"):
        for item in rec["impressions"].split("|"):
            row = {k: v for k, v in rec.items() if k != "impressions"}
            row.update(dict(zip(IMPR_FIELDS, item.split(","))))
            rows.append(row)
    flat = tmp_path / "flat.csv"
    pd.DataFrame(rows).to_csv(flat, index=False)
    out = tmp_path / "prep_flat"
    rc = subprocess.call([
        sys.executable, str(REPO_ROOT / "scripts" / "prepare_expedia_rectour.py"),
        "--raw", str(flat), "--out", str(out), "--J", "5", "--min-events", "2",
        "--max-users", "0",
    ])
    assert rc == 0
    assert len(pd.read_csv(out / "events.csv")) > 0


# --------------------------------------------------------------------------- #
# Adapter + real-slate choice sets
# --------------------------------------------------------------------------- #


def _resolved_yaml(prepared_dir: Path, tmp_path: Path) -> Path:
    import yaml

    doc = yaml.safe_load(
        (REPO_ROOT / "configs" / "datasets" / "expedia_rectour.yaml").read_text()
    )
    doc["dataset"]["events"]["path"] = str(prepared_dir / "events.csv")
    doc["dataset"]["persons"]["path"] = str(prepared_dir / "persons.csv")
    doc["dataset"]["training"]["choice_set_size"] = 6
    doc["expedia_rectour"]["impressions_path"] = str(prepared_dir / "impressions.csv")
    doc["expedia_rectour"]["properties_path"] = str(prepared_dir / "properties.csv")
    p = tmp_path / "expedia_rectour.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False))
    return p


@pytest.fixture(scope="module")
def pipeline_records(prepared_dir: Path, tmp_path_factory) -> tuple[list[dict], pd.DataFrame]:
    from src.data.adapter import YamlAdapter
    from src.data.alt_rendering import register_on_adapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import compute_hotel_aggregates, extract_extra_fields_from_row
    from src.data.expedia_slates import (
        load_alt_catalog,
        make_per_event_alt_overrides_fn,
        make_per_event_context_fn,
    )
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_brand_map, attach_train_popularity, compute_state_features,
    )
    from src.data.survey_join import join_survey

    yaml_path = _resolved_yaml(prepared_dir, tmp_path_factory.mktemp("yaml"))
    adapter = YamlAdapter(yaml_path)
    events = clean_events(adapter.load_events(), adapter.schema)
    persons_raw = adapter.load_persons()
    events = join_survey(events, persons_raw, adapter.schema)
    events = compute_state_features(events)
    events = temporal_split(events, adapter.schema)
    events = attach_train_popularity(events)
    events = attach_train_brand_map(events)
    train = events[events["split"] == "train"]
    register_on_adapter(adapter, train)
    persons_canonical = adapter.translate_z_d(persons_raw, training_events=train)

    import yaml
    doc = yaml.safe_load(yaml_path.read_text())
    extras_block = doc["dataset"]["persons"]["c_d_extra_fields"]
    customer_to_extras = {
        str(r["user_id"]): extract_extra_fields_from_row(r, extras_block)
        for r in persons_raw.to_dict("records")
    }
    for cid, agg in compute_hotel_aggregates(events).items():
        customer_to_extras.setdefault(str(cid), {}).update(agg)

    records = build_choice_sets(
        events, persons_canonical, adapter,
        seed=3, n_resamples=1, n_negatives=5,
        customer_to_extras=customer_to_extras,
        per_event_alt_overrides_fn=make_per_event_alt_overrides_fn(
            events, prepared_dir / "impressions.csv", prepared_dir / "properties.csv",
        ),
        real_slate_column="slate_prop_ids",
        alt_catalog=load_alt_catalog(prepared_dir / "properties.csv"),
        per_event_context_fn=make_per_event_context_fn(events),
    )
    return records, events


def test_records_use_the_displayed_slate_verbatim(pipeline_records) -> None:
    records, events = pipeline_records
    assert len(records) == len(events)
    for rec, slate in zip(records, events["slate_prop_ids"].astype(str)):
        assert set(rec["choice_asins"]) == set(slate.split("|"))
        assert len(rec["choice_asins"]) == 6
        assert rec["choice_asins"][rec["chosen_idx"]] == rec["chosen_asin"]
        assert rec["metadata"]["dedup_fallback"] is False


def test_chosen_position_is_shuffled_not_display_rank(pipeline_records) -> None:
    records, _ = pipeline_records
    positions = [r["chosen_idx"] for r in records]
    # The chosen item sits at slate index 0 before shuffling; after the
    # per-event permutation it must land in every position.
    assert len(set(positions)) == 6
    assert positions.count(0) < 0.6 * len(positions)


def test_alt_text_is_symmetric_and_carries_displayed_attributes(pipeline_records) -> None:
    records, _ = pipeline_records
    for rec in records[:100]:
        for alt in rec["alt_texts"]:
            assert set(alt) >= {"title", "category", "price", "popularity_rank",
                                "brand", "guest_rating", "free_cancellation",
                                "price_drop_shown"}
            assert 1.0 <= float(alt["price"]) <= 5.0
            assert alt["brand"] != "unknown_brand"
            assert alt["title"].startswith("Property #")
            assert alt["free_cancellation"] in {"yes", "no"}


def test_context_string_has_trip_line_and_no_fabricated_demographics(pipeline_records) -> None:
    records, _ = pipeline_records
    for rec in records[:50]:
        cd = rec["c_d"]
        assert "- This trip:" in cd
        assert "Income:" not in cd
        assert "Age:" not in cd
        assert "Lives in a" not in cd
        assert "Searches for lodging on Expedia" in cd


def test_slate_length_mismatch_is_loud(pipeline_records, prepared_dir: Path) -> None:
    from src.data.adapter import YamlAdapter
    from src.data.choice_sets import build_choice_sets

    _, events = pipeline_records
    bad = events.copy()
    bad.loc[bad.index[0], "slate_prop_ids"] = "|".join(
        str(bad.loc[bad.index[0], "slate_prop_ids"]).split("|")[:3]
    )
    yaml_path = REPO_ROOT / "configs" / "datasets" / "expedia_rectour.yaml"
    adapter = YamlAdapter(yaml_path)
    persons_canonical = pd.DataFrame({
        "customer_id": bad["customer_id"].unique(),
        "age_bucket": "35-44", "income_bucket": "50-100k", "household_size": 2,
        "has_kids": 0, "city_size": "large", "education": 3, "health_rating": 3,
        "risk_tolerance": 0, "purchase_frequency": 1.0, "novelty_rate": 1.0,
    })
    with pytest.raises(ValueError, match="slate of 3"):
        build_choice_sets(
            bad, persons_canonical, adapter, seed=0, n_negatives=5,
            real_slate_column="slate_prop_ids",
        )


# --------------------------------------------------------------------------- #
# Prompt + provider
# --------------------------------------------------------------------------- #


def test_hotel_anchored_prompt_dispatch_and_axes() -> None:
    from src.outcomes import generate as gen
    from src.outcomes.prompts import HOTEL_ANCHORED_AXES, build_messages_hotel_anchored

    assert len(HOTEL_ANCHORED_AXES) == 5
    alt = {"title": "Property #1", "category": "destination 2", "price": 3.0,
           "popularity_rank": "top 50%"}
    msgs = build_messages_hotel_anchored(c_d="Person profile", alt=alt, K=5)
    assert "financial, comfort, convenience, trust, experience" in msgs[0]["content"]
    assert "Property #1" in msgs[1]["content"]
    with pytest.raises(ValueError):
        build_messages_hotel_anchored(c_d="x", alt=alt, K=3)

    # generate_outcomes must route v5_hotel_anchored to the hotel builder.
    captured: dict = {}

    class _Client:
        model_id = "open/test-model"

        def generate(self, messages, *, temperature, top_p, max_tokens, seed):
            captured["messages"] = messages
            return gen.GenerationResult(
                text="\n".join(f"outcome {i} sentence" for i in range(5)),
                finish_reason="stop", model_id="open/test-model",
            )

    payload = gen.generate_outcomes(
        "c1", "a1", "Person profile", alt, K=5, seed=0,
        prompt_version="v5_hotel_anchored", client=_Client(),
    )
    assert len(payload.outcomes) == 5
    assert "one per axis" in captured["messages"][0]["content"]
    assert payload.metadata["cache_prompt_version"].startswith("v5_hotel_anchored-K5-open_test-model")


def test_openai_client_forwards_base_url_and_extra_body(monkeypatch) -> None:
    from tests.outcomes.test_openai_client import (
        _CapturingCompletions,
        _CapturingOpenAI,
        _install_fake_openai,
    )

    completions = _install_fake_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient(
        model_id="RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        base_url="http://127.0.0.1:8000/v1",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    inst = _CapturingOpenAI.last_instance
    assert inst is not None
    assert inst.init_kwargs["base_url"] == "http://127.0.0.1:8000/v1"
    assert inst.init_kwargs["api_key"] == "not-needed"  # local servers need a placeholder
    res = client.generate(
        [{"role": "user", "content": "hi"}], temperature=0.8, top_p=0.95,
        max_tokens=180, seed=1,
    )
    assert res.text
    sent = completions.calls[-1]
    assert sent["max_tokens"] == 180 and sent["temperature"] == 0.8
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert isinstance(completions, _CapturingCompletions)


def test_openai_client_reasoning_routing_disabled_behind_base_url(monkeypatch) -> None:
    """An open-weight model named like a reasoning family (o1-open) served
    locally must still get plain max_tokens + sampler knobs."""
    from tests.outcomes.test_openai_client import _install_fake_openai

    completions = _install_fake_openai(monkeypatch)
    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient(model_id="o1-open", api_key="k", base_url="http://x/v1")
    client.generate([{"role": "user", "content": "hi"}], temperature=0.5, top_p=0.9,
                    max_tokens=10, seed=0)
    sent = completions.calls[-1]
    assert "max_tokens" in sent and "max_completion_tokens" not in sent


def test_run_dataset_provider_dispatch_openai_compatible(monkeypatch, tmp_path) -> None:
    from tests.outcomes.test_openai_client import _install_fake_openai

    _install_fake_openai(monkeypatch)
    # Stub the sentence encoder import so the builder does not need the SDK.
    import types

    fake_st = types.ModuleType("sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)
    import src.outcomes.encode as enc

    class _FakeEncoder:
        def __init__(self, *a, **k):
            self.encoder_id = "fake"

    monkeypatch.setattr(enc, "SentenceTransformersEncoder", _FakeEncoder)

    import importlib
    rd = importlib.import_module("scripts.run_dataset")
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("OPENAI_MODEL", "Qwen/Qwen3-32B")
    monkeypatch.setenv("OPENAI_EXTRA_BODY", '{"chat_template_kwargs": {"enable_thinking": false}}')
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import argparse
    client, _ = rd._build_llm_and_encoder(argparse.Namespace(), {})
    assert client.model_id == "Qwen/Qwen3-32B"
    assert client._base_url == "http://127.0.0.1:8000/v1"
    assert client._extra_body == {"chat_template_kwargs": {"enable_thinking": False}}

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.3:70b")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    client, _ = rd._build_llm_and_encoder(argparse.Namespace(), {})
    assert client.model_id == "llama3.3:70b"
    assert client._base_url == "http://localhost:11434/v1"

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(SystemExit):
        rd._build_llm_and_encoder(argparse.Namespace(), {})


def test_open_weight_baseline_row_is_opt_in(monkeypatch) -> None:
    import importlib

    monkeypatch.delenv("OPENAI_COMPAT_MODEL", raising=False)
    monkeypatch.delenv("LLM_SWEEP", raising=False)
    monkeypatch.delenv("LLM_BASELINE_SKIP", raising=False)
    import src.baselines.run_all as ra
    ra = importlib.reload(ra)
    names = [n for n, _, _ in ra.BASELINE_REGISTRY]
    assert not any("Llama" in n for n in names)

    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_SWEEP", "Llama-3.3-70B-Instruct-FP8-dynamic")
    monkeypatch.setenv("LLM_BASELINE_SKIP", "LLM-SR,LaSR")
    ra = importlib.reload(ra)
    names = [n for n, _, _ in ra.BASELINE_REGISTRY]
    assert "ZeroShot-Llama-3.3-70B-Instruct-FP8-dynamic" in names
    assert "FewShot-ICL-Llama-3.3-70B-Instruct-FP8-dynamic" in names
    assert not any(n.startswith("LLM-SR") or n.startswith("LaSR") for n in names)
    monkeypatch.delenv("OPENAI_COMPAT_MODEL")
    monkeypatch.delenv("LLM_SWEEP")
    monkeypatch.delenv("LLM_BASELINE_SKIP")
    importlib.reload(ra)


def test_trip_phrase_never_trips_paraphrase_guard() -> None:
    from src.data.context_string import build_context_string
    from src.data.expedia_slates import render_trip_phrase

    phrase = render_trip_phrase({
        "checkin_date": "2021-07-14", "length_of_stay": 6, "adult_count": 2,
        "child_count": 1, "room_count": 2, "booking_window": 38, "is_mobile": 1,
        "destination_id": 3,
    })
    assert phrase == (
        "a 6 night stay, at destination 3, checking in Wednesday 14 July, "
        "for 2 adults and 1 child, in 2 rooms, searched 38 days ahead, on a mobile device"
    )
    row = {"age_bucket": "35-44", "income_bucket": "50-100k", "household_size": 3,
           "has_kids": 1, "city_size": "large", "education": 3, "health_rating": 3,
           "risk_tolerance": 0, "purchase_frequency": 1.2, "novelty_rate": 0.9}
    cd = build_context_string(
        row, suppress_fields=("age_bucket", "income_bucket", "city_size", "education",
                              "health_rating", "risk_tolerance"),
        extra_fields={"domain_verb": "Searches for lodging on Expedia",
                      "typical_star_rating": 3.6, "typical_price_tier": 4.1,
                      "typical_stay_nights": 2, "typical_party_size": 2, "typical_lead_days": 20},
        event_context=phrase,
    )
    assert "- This trip: " + phrase + "." in cd
    assert "Income:" not in cd and "Age:" not in cd
    assert "upper-mid" in cd and "a few weeks ahead" in cd


def test_external_row_carries_per_event_instrumentation(tmp_path: Path) -> None:
    """OM-LEU's leaderboard row must feed paired_significance.py."""
    import types

    from scripts.run_baselines import _external_row

    logits = np.array([[2.0, 0.5, 0.1], [0.1, 0.2, 3.0], [1.0, 1.0, 1.0]])
    c_star = np.array([0, 2, 1])
    npz = tmp_path / "test_logits.npz"
    np.savez(npz, logits=logits.astype(np.float32), c_star=c_star, n_params=np.int64(7))
    batch = types.SimpleNamespace(
        n_events=3, chosen_indices=c_star, customer_ids=["a", "a", "b"],
    )
    row = _external_row("OM-LEU", str(npz), batch, train_n_events=10)
    assert row["status"] == "ok"
    assert len(row["per_event_nll"]) == 3
    assert abs(float(np.mean(row["per_event_nll"])) - row["test_nll"]) < 1e-6
    assert row["per_event_topk_correct"] == [True, True, False]
    assert set(row["per_customer_nll"]) == {"a", "b"}


def test_dominant_attribute_report_lists_every_head_even_if_never_dominant() -> None:
    import torch

    from src.eval.interpret import dominant_attribute_report, head_naming_report
    from src.model.po_leu import POLEUIntermediates

    B, J, K, M = 6, 3, 2, 5
    torch.manual_seed(0)
    A = torch.rand(B, J, K, M) * 0.1
    A[..., 2] += 5.0  # head 2 dominates every event
    w = torch.full((B, M), 1.0 / M)
    inter = POLEUIntermediates(A=A, w=w, U=A.sum(-1), S=torch.full((B, J, K), 0.5),
                               V=A.sum(-1).sum(-1), V_residual=None, V_total=None)
    logits = torch.randn(B, J)
    c_star = torch.zeros(B, dtype=torch.int64)
    rep = dominant_attribute_report(logits, c_star, inter)
    assert set(rep["n_by_attribute"]) == {f"m{i}" for i in range(M)}
    assert rep["n_by_attribute"]["m2"] == B
    for i in (0, 1, 3, 4):
        assert rep["n_by_attribute"][f"m{i}"] == 0
        assert all(v is None for v in rep["by_dominant_attribute"][f"m{i}"].values())
    assert rep["by_dominant_attribute"]["m2"]["nll"] is not None

    # Head naming: duplicate strings collapse to one entry each.
    outcomes = [[["same text", "other text"] for _ in range(J)] for _ in range(B)]
    named = head_naming_report(outcomes, inter, top_n=10)
    for m in range(M):
        texts = [r["outcome"] for r in named["top_outcomes_per_head"][f"m{m}"]]
        assert len(texts) == len(set(texts)) == 2
