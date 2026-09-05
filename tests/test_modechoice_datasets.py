"""Optima + LPMC prepare scripts and the generic mode-choice slate module.

Hermetic: tiny raw files in each dataset's documented column layout are
written per module; the full data-layer chain runs on them.
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

OPTIMA_COLS = ["ID", "DestAct", "NbTransf", "TimePT", "WalkingTimePT", "WaitingTimePT", "CostPT",
               "CostCar", "TimeCar", "NbHousehold", "NbChild", "NbCar", "Income", "Gender",
               "BirthYear", "FamilSitu", "OccupStat", "Education", "HalfFareST", "LineRelST",
               "GenAbST", "AreaRelST", "OtherST", "CarAvail", "MarginalCostPT", "CostCarCHF",
               "TripPurpose", "TypeCommune", "UrbRur", "LangCode", "NbTrajects", "Region",
               "distance_km", "Choice", "age"]

LPMC_COLS = ["trip_id", "household_id", "person_n", "trip_n", "travel_mode", "purpose", "fueltype",
             "faretype", "bus_scale", "survey_year", "travel_year", "travel_month", "travel_date",
             "day_of_week", "start_time", "age", "female", "driving_license", "car_ownership",
             "distance", "dur_walking", "dur_cycling", "dur_pt_access", "dur_pt_rail", "dur_pt_bus",
             "dur_pt_int", "pt_interchanges", "dur_driving", "cost_transit", "cost_driving_fuel",
             "cost_driving_ccharge", "driving_traffic_percent"]


def _write_optima(path: Path, n: int = 60, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rid = 10000 + i // 2 if i % 3 == 0 else 20000 + i  # a few respondents with 2 loops
        choice = int(rng.choice([0, 1, 2, -1], p=[0.3, 0.55, 0.1, 0.05]))
        rows.append([rid, int(rng.integers(1, 12)), int(rng.integers(0, 5)), int(rng.integers(20, 200)),
                     int(rng.integers(0, 40)), int(rng.integers(0, 30)), int(rng.integers(2, 40)),
                     round(float(rng.uniform(0, 30)), 2), int(rng.integers(10, 120)), int(rng.choice([-1, 1, 2, 3, 4])),
                     int(rng.choice([-1, 0, 1, 2])), int(rng.choice([-1, 0, 1, 2])), int(rng.choice([-1, 1, 2, 3, 4, 5, 6])),
                     int(rng.choice([-1, 1, 2])), int(rng.integers(1940, 1995)), int(rng.choice([-1, 1, 2, 3])),
                     int(rng.choice([-1, 1, 2, 9])), int(rng.choice([-1, 2, 3, 6, 7])), int(rng.choice([1, 2])),
                     2, int(rng.choice([1, 2], p=[0.1, 0.9])), 2, 2, int(rng.choice([-1, 1, 2, 3])),
                     round(float(rng.uniform(0, 20)), 2), round(float(rng.uniform(0, 30)), 2),
                     int(rng.choice([-1, 1, 2, 3])), int(rng.integers(1, 10)), int(rng.choice([1, 2])),
                     int(rng.choice([1, 2])), int(rng.integers(1, 6)), int(rng.integers(1, 9)),
                     round(float(rng.uniform(1, 80)), 1), choice, int(rng.choice([-1, 25, 40, 60]))])
    pd.DataFrame(rows, columns=OPTIMA_COLS).to_csv(path, sep="\t", index=False)
    return path


def _write_lpmc(path: Path, n_persons: int = 30, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    tid = 0
    for p in range(n_persons):
        hh, pn = p // 2, p % 2
        n_trips = int(rng.integers(2, 9))
        age, fem, lic, own = int(rng.integers(18, 80)), int(rng.integers(0, 2)), int(rng.integers(0, 2)), int(rng.integers(0, 3))
        for t in range(n_trips):
            month, day = int(rng.integers(1, 13)), int(rng.integers(1, 28))
            rows.append([tid, hh, pn, t, int(rng.choice([1, 2, 3, 4], p=[0.18, 0.03, 0.35, 0.44])),
                         int(rng.integers(1, 6)), 1, int(rng.integers(1, 6)), 1.0, 1, 2012 + int(rng.integers(0, 3)),
                         month, day, int(rng.integers(1, 8)), round(float(rng.uniform(5, 23)), 2), age, fem, lic, own,
                         int(rng.integers(300, 15000)), round(float(rng.uniform(0.05, 3)), 3),
                         round(float(rng.uniform(0.02, 1)), 3), round(float(rng.uniform(0, 0.4)), 3),
                         round(float(rng.uniform(0, 0.5)), 3), round(float(rng.uniform(0, 0.5)), 3),
                         round(float(rng.uniform(0, 0.2)), 3), int(rng.integers(0, 3)), round(float(rng.uniform(0.05, 1)), 3),
                         round(float(rng.uniform(0, 5)), 2), round(float(rng.uniform(0, 3)), 2),
                         float(rng.choice([0.0, 10.5])), round(float(rng.uniform(0, 1)), 3)])
            tid += 1
    pd.DataFrame(rows, columns=LPMC_COLS).to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(scope="module")
def optima_dir(tmp_path_factory) -> Path:
    raw = _write_optima(tmp_path_factory.mktemp("optima_raw") / "optima.dat")
    out = tmp_path_factory.mktemp("optima_prep")
    assert subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_optima.py"),
                            "--raw", str(raw), "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def lpmc_dir(tmp_path_factory) -> Path:
    raw = _write_lpmc(tmp_path_factory.mktemp("lpmc_raw") / "lpmc.dat")
    out = tmp_path_factory.mktemp("lpmc_prep")
    assert subprocess.call([sys.executable, str(REPO_ROOT / "scripts" / "prepare_lpmc.py"),
                            "--raw", str(raw), "--out", str(out)]) == 0
    return out


def test_optima_prepare_layout(optima_dir: Path) -> None:
    ev = pd.read_csv(optima_dir / "events.csv")
    imp = pd.read_csv(optima_dir / "impressions.csv", keep_default_na=False)
    persons = pd.read_csv(optima_dir / "persons.csv")
    s = json.loads((optima_dir / "prepare_summary.json").read_text())
    assert s["n_events"] == len(ev) and (ev["slate_alt_ids"] == "pt|car|soft").all()
    assert set(ev["alt_id"]) <= {"pt", "car", "soft"}
    assert len(imp) == 3 * len(ev) and set(imp["alt_id"]) == {"pt", "car", "soft"}
    assert (imp.loc[imp["alt_id"] == "soft", "price"].astype(float) == 0).all()
    assert set(persons["age_bucket"]) <= {"18-24", "25-34", "35-44", "45-54", "55-64", "65+"}
    assert set(persons["city_size"]) <= {"rural", "small", "medium"}
    assert ev["trip_phrase"].str.contains("loop").all()
    assert "education" not in " ".join(ev["trip_phrase"]).lower()


def test_lpmc_prepare_layout(lpmc_dir: Path) -> None:
    ev = pd.read_csv(lpmc_dir / "events.csv")
    imp = pd.read_csv(lpmc_dir / "impressions.csv", keep_default_na=False)
    persons = pd.read_csv(lpmc_dir / "persons.csv")
    assert (ev["slate_alt_ids"] == "walk|cycle|pt|drive").all()
    assert len(imp) == 4 * len(ev)
    assert pd.to_datetime(ev["timestamp"]).notna().all()
    drive = imp[imp["alt_id"] == "drive"]
    assert drive["journey_details"].str.contains("fuel GBP").all()
    assert persons["profile_lines"].str.contains("licence").all()
    assert "education" not in " ".join(ev["purpose_label"].astype(str)).lower()


def _run_chain(adapter: str, prepared: Path, split_mode: str, min_events: int):
    import yaml

    from src.data.adapter import YamlAdapter
    from src.data.alt_rendering import register_on_adapter
    from src.data.choice_sets import build_choice_sets
    from src.data.clean import clean_events
    from src.data.context_string import extract_extra_fields_from_row
    from src.data.expedia_slates import load_alt_catalog
    from src.data.modechoice_slates import make_per_event_alt_overrides_fn, make_per_event_context_fn
    from src.data.split import cold_start_split, temporal_split
    from src.data.state_features import attach_train_brand_map, attach_train_popularity, compute_state_features
    from src.data.survey_join import join_survey

    doc = yaml.safe_load((REPO_ROOT / "configs" / "datasets" / f"{adapter}.yaml").read_text())
    doc["dataset"]["events"]["path"] = str(prepared / "events.csv")
    doc["dataset"]["persons"]["path"] = str(prepared / "persons.csv")
    yp = prepared / "resolved.yaml"
    yp.write_text(yaml.safe_dump(doc, sort_keys=False))
    ad = YamlAdapter(yp)
    events = clean_events(ad.load_events(), ad.schema)
    persons_raw = ad.load_persons()
    events = join_survey(events, persons_raw, ad.schema)
    events = compute_state_features(events)
    counts = events.groupby("customer_id").size()
    events = events[events["customer_id"].isin(counts[counts >= min_events].index)].reset_index(drop=True)
    events = cold_start_split(events, schema=ad.schema, seed=1) if split_mode == "cold_start" else temporal_split(events, ad.schema)
    events = attach_train_popularity(events)
    events = attach_train_brand_map(events)
    train = events[events["split"] == "train"]
    register_on_adapter(ad, train)
    pc = ad.translate_z_d(persons_raw, training_events=train)
    events = events[events["customer_id"].isin(set(pc["customer_id"]))].reset_index(drop=True)
    extras = {str(r[ad.schema.persons_id_column]): extract_extra_fields_from_row(r, doc["dataset"]["persons"]["c_d_extra_fields"])
              for r in persons_raw.to_dict("records")}
    mc = doc["modechoice"]
    recs = build_choice_sets(
        events, pc, ad, seed=1, n_negatives=int(ad.schema.choice_set_size) - 1,
        customer_to_extras=extras,
        per_event_alt_overrides_fn=make_per_event_alt_overrides_fn(
            events, prepared / "impressions.csv", prepared / "properties.csv",
            event_id_column=mc["event_id_column"], currency=mc["currency"]),
        real_slate_column=mc["slate_column"],
        alt_catalog=load_alt_catalog(prepared / "properties.csv"),
        per_event_context_fn=make_per_event_context_fn(events, phrase_column=mc["phrase_column"]),
    )
    return recs, events


def test_optima_chain_cold_start(optima_dir: Path) -> None:
    recs, events = _run_chain("optima", optima_dir, "cold_start", 1)
    assert len(recs) == len(events) > 0
    splits = {r["split"] for r in recs}
    assert {"train", "val", "test"} <= splits
    # cold start: val/test customers never appear in train
    by = {}
    for r in recs:
        by.setdefault(r["customer_id"], set()).add(r["split"])
    assert all(len(s) == 1 for s in by.values())
    # Held-out customers have no train events; their derived covariates
    # (purchase_frequency, novelty_rate) must still be finite, and so must
    # every z_d (the NaN used to poison the standardization for everyone).
    assert all(np.isfinite(r["z_d"]).all() for r in recs)
    for r in recs[:40]:
        assert set(r["choice_asins"]) == {"pt", "car", "soft"}
        assert "- This trip:" in r["c_d"] and "CHF" in r["c_d"]
        cats = {a["category"] for a in r["alt_texts"]}
        assert cats == {r["category"]}
        for a in r["alt_texts"]:
            assert a["currency"] == "CHF" and "travel_time" in a


def test_lpmc_chain_temporal(lpmc_dir: Path) -> None:
    recs, events = _run_chain("lpmc", lpmc_dir, "temporal", 5)
    assert len(recs) == len(events) > 0
    for r in recs[:40]:
        assert set(r["choice_asins"]) == {"walk", "cycle", "pt", "drive"}
        assert "- This trip:" in r["c_d"]
        assert "Income:" not in r["c_d"]  # LPMC has no income; constant is suppressed
        drive = r["alt_texts"][r["choice_asins"].index("drive")]
        assert drive["currency"] == "GBP" and "journey_details" in drive
        walk = r["alt_texts"][r["choice_asins"].index("walk")]
        assert float(walk["price"]) == 0.0 and "journey_details" not in walk  # empty cells skipped


def test_modechoice_prompt_dispatch() -> None:
    from src.outcomes import generate as gen

    captured = {}

    class _C:
        model_id = "open/x"

        def generate(self, messages, **kw):
            captured["m"] = messages
            return gen.GenerationResult(text="\n".join(["s"] * 5), finish_reason="stop", model_id="open/x")

    alt = {"title": "Driving", "category": "commute", "price": 3.2, "currency": "GBP",
           "popularity_rank": "top 50%", "travel_time": "25 minutes"}
    gen.generate_outcomes("c", "drive", "Person profile", alt, K=5, seed=0,
                          prompt_version="v7_modechoice_anchored", client=_C())
    sysm, user = captured["m"][0]["content"], captured["m"][1]["content"]
    assert "how to make a specific trip" in sysm and "financial, time, comfort, convenience, reliability" in sysm
    assert "- Price: GBP 3.20" in user and "currency" not in user
