"""Tests for the Swissmetro onboarding (prepare -> adapter -> real slates).

Hermetic: a tiny synthetic ``swissmetro.dat`` in the official column layout
is written per module; no network, no LLM.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

COLS = ["GROUP", "SURVEY", "SP", "ID", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE",
        "AGE", "MALE", "INCOME", "GA", "ORIGIN", "DEST", "TRAIN_AV", "CAR_AV", "SM_AV",
        "TRAIN_TT", "TRAIN_CO", "TRAIN_HE", "SM_TT", "SM_CO", "SM_HE", "SM_SEATS",
        "CAR_TT", "CAR_CO", "CHOICE"]


def _write_raw(path: Path, *, n_resp: int = 40, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for rid in range(1, n_resp + 1):
        car_av = 0 if rid % 8 == 0 else 1          # some non-car-owners
        ga = 1 if rid % 10 == 0 else 0
        age = int(rng.integers(1, 6)); inc = int(rng.integers(0, 5)); male = int(rng.integers(0, 2))
        first = int(rng.integers(0, 2)); grp = int(rng.choice([2, 3])); purpose = int(rng.integers(1, 5))
        who = int(rng.integers(0, 4)); lug = int(rng.choice([0, 1, 3])); ticket = int(rng.integers(1, 8))
        o, d = int(rng.integers(1, 27)), int(rng.integers(1, 27))
        for k in range(9):
            tt_tr, tt_sm, tt_car = rng.integers(60, 240), rng.integers(30, 120), rng.integers(60, 240)
            co_tr = 5040 if ga else rng.integers(20, 150)
            co_sm, co_car = rng.integers(20, 200), rng.integers(10, 150)
            u = np.array([-0.01 * tt_tr - 0.01 * (0 if ga else co_tr), -0.01 * tt_sm - 0.01 * co_sm + 0.5,
                          -0.01 * tt_car - 0.01 * co_car]) + rng.gumbel(size=3)
            if not car_av:
                u[2] = -1e9
            choice = int(np.argmax(u)) + 1
            if rid == 1 and k == 0:
                choice = 0  # one unknown choice row must be dropped
            rows.append([grp, 0, 1, rid, purpose, first, ticket, who, lug, age, male, inc, ga, o, d,
                         1, car_av, 1, tt_tr, co_tr, int(rng.choice([30, 60, 120])), tt_sm, co_sm,
                         int(rng.choice([10, 20, 30])), int(rng.integers(0, 2)), tt_car, co_car, choice])
    pd.DataFrame(rows, columns=COLS).to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(scope="module")
def prepared_dir(tmp_path_factory) -> Path:
    raw = _write_raw(tmp_path_factory.mktemp("raw") / "swissmetro.dat")
    out = tmp_path_factory.mktemp("prepared")
    rc = subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_swissmetro.py"),
                          "--raw", str(raw), "--out", str(out)])
    assert rc == 0
    return out


def test_prepare_outputs_and_availability_filter(prepared_dir: Path) -> None:
    s = json.loads((prepared_dir / "prepare_summary.json").read_text())
    ev = pd.read_csv(prepared_dir / "events.csv")
    persons = pd.read_csv(prepared_dir / "persons.csv")
    imp = pd.read_csv(prepared_dir / "impressions.csv")
    props = pd.read_csv(prepared_dir / "properties.csv")
    # 40 respondents, 5 without a car dropped entirely, one unknown-choice row dropped.
    assert s["n_respondents"] == 35
    assert set(ev.groupby("respondent_id").size()) <= {8, 9}
    assert set(ev["alt_id"]) <= {"train", "sm", "car"}
    assert (ev["slate_alt_ids"] == "train|sm|car").all()
    assert ev["respondent_id"].dtype == object and ev["respondent_id"].str.startswith("r").all()
    assert persons["respondent_id"].dtype == object
    assert set(persons["age_class"]) <= {"under_25", "25_to_39", "39_to_54", "54_to_65", "over_65", "unknown"}
    assert set(persons["gender_label"]) <= {"Male", "Female"}
    assert len(props) == 3 and set(props["prop_id"]) == {"train", "sm", "car"}
    assert len(imp) == 3 * len(ev)
    # GA holders: rail costs are zero (Biogeme convention), car cost untouched.
    ga_ids = set(persons.loc[persons["ga"] == 1, "respondent_id"])
    ga_scen = set(ev.loc[ev["respondent_id"].isin(ga_ids), "scenario_id"])
    rail = imp[imp["scenario_id"].isin(ga_scen) & imp["alt_id"].isin(["train", "sm"])]
    assert (rail["cost_chf"] == 0).all()
    assert "holds a GA annual season ticket" in "|".join(persons.loc[persons["ga"] == 1, "profile_lines"].astype(str))
    # Pseudo-dates are 7 days apart within a respondent.
    d = pd.to_datetime(ev[ev["respondent_id"] == ev["respondent_id"].iloc[0]]["pseudo_date"])
    assert (d.diff().dropna().dt.days == 7).all()


@pytest.fixture(scope="module")
def records(prepared_dir: Path, tmp_path_factory):
    import yaml

    from src.data.adapter import YamlAdapter
    from src.data.alt_rendering import register_on_adapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import extract_extra_fields_from_row
    from src.data.expedia_slates import load_alt_catalog
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_brand_map, attach_train_popularity, compute_state_features,
    )
    from src.data.survey_join import join_survey
    from src.data.swissmetro_slates import (
        make_per_event_alt_overrides_fn, make_per_event_context_fn,
    )

    doc = yaml.safe_load((REPO_ROOT / "configs" / "datasets" / "swissmetro.yaml").read_text())
    doc["dataset"]["events"]["path"] = str(prepared_dir / "events.csv")
    doc["dataset"]["persons"]["path"] = str(prepared_dir / "persons.csv")
    yp = tmp_path_factory.mktemp("yaml") / "swissmetro.yaml"
    yp.write_text(yaml.safe_dump(doc, sort_keys=False))
    adapter = YamlAdapter(yp)
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
    events = events[events["customer_id"].isin(set(persons_canonical["customer_id"]))].reset_index(drop=True)
    extras_block = doc["dataset"]["persons"]["c_d_extra_fields"]
    customer_to_extras = {
        str(r["respondent_id"]): extract_extra_fields_from_row(r, extras_block)
        for r in persons_raw.to_dict("records")
    }
    from src.data.context_string import compute_customer_aggregates
    for cid, agg in compute_customer_aggregates(events, train_only=True).items():
        customer_to_extras.setdefault(str(cid), {}).update(agg)
    recs = build_choice_sets(
        events, persons_canonical, adapter, seed=1, n_negatives=2,
        customer_to_extras=customer_to_extras,
        per_event_alt_overrides_fn=make_per_event_alt_overrides_fn(
            events, prepared_dir / "impressions.csv", prepared_dir / "properties.csv"),
        real_slate_column="slate_alt_ids",
        alt_catalog=load_alt_catalog(prepared_dir / "properties.csv"),
        per_event_context_fn=make_per_event_context_fn(events),
    )
    return recs, events


def test_records_are_three_mode_slates_with_scenario_attributes(records) -> None:
    recs, events = records
    assert len(recs) == len(events)
    for r in recs:
        assert set(r["choice_asins"]) == {"train", "sm", "car"}
        assert r["choice_asins"][r["chosen_idx"]] == r["chosen_asin"]
        for alt in r["alt_texts"]:
            assert alt["brand"] in {"regular rail", "maglev rail", "private car"}
            assert "travel_time" in alt and "departures" in alt
            assert float(alt["price"]) >= 0.0
    # Chosen position is shuffled, not fixed.
    assert len({r["chosen_idx"] for r in recs}) == 3
    # No chosen-vs-non-chosen asymmetry in any rendered field: everything
    # per-alternative must be explainable by (scenario, mode) alone. In
    # particular the event-level ``category`` must be identical across the
    # slate (it used to leak the chosen mode's first-seen purpose).
    for r in recs:
        cats = {alt["category"] for alt in r["alt_texts"]}
        assert len(cats) == 1 and cats == {r["category"]}
        titles = {alt["title"] for alt in r["alt_texts"]}
        assert len(titles) == 3


def test_context_string_carries_trip_profile_and_gender(records) -> None:
    recs, _ = records
    joined = "\n".join(r["c_d"] for r in recs)
    assert "- This trip:" in joined
    assert "Income:" in joined and "Age:" in joined  # real covariates render
    assert ", as a male" in joined or ", as a female" in joined
    assert "currently makes this kind of trip by" in joined.lower()
    assert "Makes intercity trips" in joined
    assert "- This trip: a " in joined  # purpose survives the category rename
    assert "CHF" in joined and "$" not in joined
    assert "Mode chosen most often so far:" in joined
    assert "purchase" not in joined.lower()


def test_travel_prompt_dispatch() -> None:
    from src.outcomes import generate as gen
    from src.outcomes.prompts import TRAVEL_ANCHORED_AXES

    alt = {"title": "Private car", "category": "business", "price": 84.0,
           "popularity_rank": "top 50%", "travel_time": "2 hours", "departures": "whenever you choose to leave"}
    captured = {}

    class _C:
        model_id = "open/x"

        def generate(self, messages, **kw):
            captured["m"] = messages
            return gen.GenerationResult(text="\n".join(["s"] * 5), finish_reason="stop", model_id="open/x")

    gen.generate_outcomes("c", "car", "Person profile", alt, K=5, seed=0,
                          prompt_version="v6_travel_anchored", client=_C())
    assert "financial, time, comfort, convenience, reliability" in captured["m"][0]["content"]
    assert "travel_time: 2 hours" in captured["m"][1]["content"]
    assert len(TRAVEL_ANCHORED_AXES) == 5


def test_llm_rankers_get_letters_sized_to_J() -> None:
    from src.baselines._llm_ranker_common import DEFAULT_LETTERS
    from src.baselines.few_shot_icl_ranker import FewShotICLRanker
    from src.baselines.run_all import _auto_letters
    from src.baselines.zero_shot_claude_ranker import ZeroShotClaudeRanker

    for cls in (ZeroShotClaudeRanker, FewShotICLRanker):
        kw: dict = {}
        _auto_letters(cls, 3, kw)
        assert kw["letters"] == tuple(DEFAULT_LETTERS[:3])
        kw = {}
        _auto_letters(cls, len(DEFAULT_LETTERS), kw)
        assert "letters" not in kw  # default J: leave the class default alone
        kw = {"letters": ("X", "Y", "Z")}
        _auto_letters(cls, 3, kw)
        assert kw["letters"] == ("X", "Y", "Z")  # explicit wins

    class NoLetters:
        def __init__(self, seed: int = 0) -> None: ...

    kw = {}
    _auto_letters(NoLetters, 3, kw)
    assert kw == {}
