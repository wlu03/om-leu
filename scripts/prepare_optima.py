"""Prepare the Optima (Swiss CarPostal) revealed-preference mode-choice survey.

Input: ``optima.dat`` (whitespace-separated; Bierlaire, CS_OptimaDescription
2018). 1,906 home-based trip loops from 1,763 respondents living in
low-density Swiss areas, 2009-2010. ``Choice``: 0 public transport, 1
private motorised modes, 2 soft modes (walk / bike); -1 missing (dropped).

Outputs (``--out``, default ``<repo>/optima/data/``): the generic
mode-choice layout consumed by ``src/data/modechoice_slates.py``:

* ``events.csv``      — one row per loop: ``respondent_id, event_id,
                        pseudo_date, alt_id, alt_label, cost_chf, purpose_label,
                        trip_phrase, slate_alt_ids``.
* ``persons.csv``     — one row per respondent (age / income / gender /
                        household / commune / profile lines).
* ``impressions.csv`` — one row per (loop, mode) with ``price`` and
                        human-readable attribute strings.
* ``properties.csv``  — the three modes.

Conventions
-----------
* Most respondents report a single loop, so the per-respondent temporal
  split cannot hold out anything: run with ``--split-mode cold_start``
  (val/test respondents unseen in training) and
  ``--min-events-per-customer 1``. ``pseudo_date`` (7-day steps within a
  respondent) only orders the few multi-loop respondents.
* Public-transport price is ``MarginalCostPT`` (cost after the
  respondent's season tickets; 0 for GA holders), car price is
  ``CostCarCHF`` (fuel), soft modes cost 0. All three modes are treated as
  available (the reference Biogeme specification does the same); car
  availability is rendered into the context string instead.
* ``-1`` codes are missing: imputed to the modal class for the canonical
  z_d buckets, and the corresponding profile line is omitted.

Usage
-----
    python scripts/prepare_optima.py --raw optima/data/raw/optima.dat [--out optima/data]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("prepare_optima")

ALT_IDS: tuple[str, ...] = ("pt", "car", "soft")
CHOICE_TO_ALT: dict[int, str] = {0: "pt", 1: "car", 2: "soft"}
ALT_LABELS: dict[str, str] = {
    "pt": "Public transport (train, bus, tram)",
    "car": "Private motorised transport (car or motorbike)",
    "soft": "Soft modes (walking or cycling)",
}
ALT_BRAND: dict[str, str] = {"pt": "public transport", "car": "private vehicle", "soft": "active travel"}

PURPOSE_CATEGORY: dict[int, str] = {1: "work", 2: "work and leisure", 3: "leisure"}
DEST_ACT: dict[int, str] = {
    1: "work", 2: "a professional trip", 3: "studying", 4: "shopping", 5: "an activity at home",
    6: "eating or drinking out", 7: "personal business", 8: "driving someone",
    9: "a cultural or sports activity", 10: "going out with friends", 11: "another activity",
}
REGION: dict[int, str] = {
    1: "the Vaud region", 2: "the Valais", 3: "the Delemont area", 4: "the Bern region",
    5: "the Basel, Aargau and Olten region", 6: "the Zurich region",
    7: "eastern Switzerland", 8: "Graubunden",
}
COMMUNE: dict[int, str] = {
    1: "a town centre", 2: "a suburban commune", 3: "a high-income commune", 4: "a periurban commune",
    5: "a touristic commune", 6: "an industrial or tertiary commune", 7: "a rural commuting commune",
    8: "an agricultural and mixed commune", 9: "an agricultural commune",
}
COMMUNE_CITY_SIZE: dict[int, str] = {1: "medium", 2: "small", 3: "small", 4: "small"}
OCCUP: dict[int, str] = {
    1: "works full time", 2: "works part time", 3: "is looking for a job", 4: "works occasionally",
    5: "has no paid job", 6: "is a homemaker", 7: "is on disability leave", 8: "is a student",
    9: "is retired",
}
FAMILY: dict[int, str] = {
    1: "lives alone", 2: "lives as a couple without children", 3: "lives as a couple with children",
    4: "is a single parent", 5: "lives in a shared flat", 6: "lives with their parents",
    7: "has another living arrangement",
}
CAR_AVAIL: dict[int, str] = {
    1: "always has a car available", 2: "sometimes has a car available", 3: "never has a car available",
}
INCOME_MAP: dict[int, str] = {1: "25-50k", 2: "25-50k", 3: "50-100k", 4: "50-100k", 5: "100-150k", 6: "150k+"}
EDU_MAP: dict[int, int] = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 5, 8: 5}


def _fmt_minutes(m: float) -> str:
    if not np.isfinite(m) or m < 0:
        return ""
    m = int(round(m))
    if m < 60:
        return f"{m} minutes"
    h, r = divmod(m, 60)
    return f"{h} hour{'s' if h != 1 else ''}" if r == 0 else f"{h} h {r:02d} min"


def _age_bucket(age: float) -> str:
    if not np.isfinite(age) or age < 0:
        return "45-54"  # modal class
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def _hh_bucket(n: float) -> str:
    if not np.isfinite(n) or n < 1:
        return "pair"
    return {1: "solo", 2: "pair", 3: "three", 4: "four"}.get(int(n), "five_plus")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "optima" / "data")
    p.add_argument("--log-level", default="INFO")
    return p


def prepare(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.raw, sep=r"\s+")
    n_raw = len(df)
    df = df[df["Choice"].isin([0, 1, 2])].copy()
    df["respondent_id"] = "o" + df["ID"].astype(int).astype(str)
    df["loop_idx"] = df.groupby("ID").cumcount() + 1
    df["event_id"] = df["respondent_id"] + "_l" + df["loop_idx"].astype(str)
    base = pd.Timestamp("2010-01-01")
    df["pseudo_date"] = [
        (base + pd.to_timedelta(7 * (int(i) - 1), unit="D")).strftime("%Y-%m-%d") for i in df["loop_idx"]
    ]
    df["alt_id"] = df["Choice"].astype(int).map(CHOICE_TO_ALT)
    pt_cost = pd.to_numeric(df["MarginalCostPT"], errors="coerce").clip(lower=0).fillna(0.0)
    car_cost = pd.to_numeric(df["CostCarCHF"], errors="coerce").clip(lower=0).fillna(0.0)
    cost = np.select([df["alt_id"] == "pt", df["alt_id"] == "car"], [pt_cost, car_cost], default=0.0)

    def _purpose(v) -> str:
        return PURPOSE_CATEGORY.get(int(v), "unknown") if pd.notna(v) else "unknown"

    def _phrase(r) -> str:
        bits = []
        purpose = PURPOSE_CATEGORY.get(int(r["TripPurpose"]), "")
        if purpose:
            bits.append(f"a {purpose} loop")
        else:
            bits.append("a loop of trips")
        nt = r.get("NbTrajects")
        if pd.notna(nt) and nt > 0:
            bits[-1] += f" of {int(nt)} trip{'s' if int(nt) != 1 else ''} starting and ending at home"
        da = DEST_ACT.get(int(r["DestAct"]), "") if pd.notna(r["DestAct"]) else ""
        if da:
            bits.append(f"mainly for {da}")
        dist = r.get("distance_km")
        if pd.notna(dist) and dist > 0:
            bits.append(f"about {float(dist):.0f} km in total")
        com = COMMUNE.get(int(r["TypeCommune"]), "")
        reg = REGION.get(int(r["Region"]), "")
        if com or reg:
            bits.append("from " + " in ".join(x for x in (com, reg) if x))
        return ", ".join(bits)

    ev = pd.DataFrame({
        "respondent_id": df["respondent_id"],
        "event_id": df["event_id"],
        "pseudo_date": df["pseudo_date"],
        "alt_id": df["alt_id"],
        "alt_label": df["alt_id"].map(ALT_LABELS),
        "cost_chf": cost.astype(float),
        "purpose_label": df["TripPurpose"].map(_purpose),
        "trip_phrase": [_phrase(r) for r in df.to_dict("records")],
        "slate_alt_ids": "|".join(ALT_IDS),
    })

    imp_rows: list[dict] = []
    for r in df.to_dict("records"):
        eid = r["event_id"]
        walk, wait = float(r["WalkingTimePT"]), float(r["WaitingTimePT"])
        ptc = max(float(r["MarginalCostPT"]), 0.0)
        imp_rows.append({
            "event_id": eid, "alt_id": "pt", "price": ptc,
            "travel_time": _fmt_minutes(float(r["TimePT"])),
            "walking_and_waiting": f"{int(round(walk))} min walking, {int(round(wait))} min waiting",
            "transfers": str(int(r["NbTransf"])),
            "fare_note": ("covered by a season ticket" if ptc <= 0
                          else f"after season-ticket discounts (full fare CHF {float(r['CostPT']):.0f})"),
        })
        imp_rows.append({
            "event_id": eid, "alt_id": "car", "price": max(float(r["CostCarCHF"]), 0.0),
            "travel_time": _fmt_minutes(float(r["TimeCar"])), "walking_and_waiting": "",
            "transfers": "", "fare_note": "fuel cost for the whole loop",
        })
        dist = float(r["distance_km"]) if pd.notna(r["distance_km"]) else float("nan")
        imp_rows.append({
            "event_id": eid, "alt_id": "soft", "price": 0.0,
            "travel_time": (f"about {dist:.0f} km on foot or by bike" if np.isfinite(dist) and dist > 0 else ""),
            "walking_and_waiting": "", "transfers": "", "fare_note": "no fare",
        })
    imp = pd.DataFrame(imp_rows)

    props = pd.DataFrame({
        "prop_id": list(ALT_IDS),
        "title": [ALT_LABELS[a] for a in ALT_IDS],
        "category": ["travel mode"] * 3,
        "price": [float(imp.loc[imp["alt_id"] == a, "price"].median()) for a in ALT_IDS],
        "brand": [ALT_BRAND[a] for a in ALT_IDS],
    })

    p_rows: list[dict] = []
    for rid, g in df.groupby("respondent_id", sort=True):
        f = g.iloc[0]
        lines: list[str] = []
        ca = CAR_AVAIL.get(int(f["CarAvail"]), "")
        if ca:
            lines.append(ca)
        tickets = []
        if int(f["GenAbST"]) == 1:
            tickets.append("a GA full Swiss rail season ticket")
        if int(f["HalfFareST"]) == 1:
            tickets.append("a half-fare travelcard")
        if int(f["LineRelST"]) == 1 or int(f["AreaRelST"]) == 1:
            tickets.append("a line or area season ticket")
        if tickets:
            lines.append("holds " + " and ".join(tickets))
        occ = OCCUP.get(int(f["OccupStat"]), "")
        if occ:
            lines.append(occ)
        fam = FAMILY.get(int(f["FamilSitu"]), "")
        if fam:
            lines.append(fam)
        age = float(f["age"]) if pd.notna(f["age"]) else -1.0
        inc = int(f["Income"]) if pd.notna(f["Income"]) else -1
        p_rows.append({
            "respondent_id": rid,
            "age_bucket": _age_bucket(age),
            "income_bucket": INCOME_MAP.get(inc, "50-100k"),
            "gender_label": {1: "Male", 2: "Female"}.get(int(f["Gender"]), ""),
            "household_bucket": _hh_bucket(float(f["NbHousehold"])),
            "has_kids": int(pd.notna(f["NbChild"]) and f["NbChild"] > 0),
            "city_size": ("rural" if int(f["UrbRur"]) == 1 else COMMUNE_CITY_SIZE.get(int(f["TypeCommune"]), "small")),
            "education": EDU_MAP.get(int(f["Education"]), 3),
            "profile_lines": "|".join(lines[:4]),
        })
    persons = pd.DataFrame(p_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "events.csv", index=False)
    persons.to_csv(out / "persons.csv", index=False)
    imp.to_csv(out / "impressions.csv", index=False)
    props.to_csv(out / "properties.csv", index=False)
    summary = {
        "raw": str(args.raw), "n_rows_raw": int(n_raw), "n_events": int(len(ev)),
        "n_respondents": int(ev["respondent_id"].nunique()),
        "events_per_respondent": ev.groupby("respondent_id").size().value_counts().to_dict(),
        "choice_shares": ev["alt_id"].value_counts(normalize=True).round(4).to_dict(),
        "purpose_shares": ev["purpose_label"].value_counts(normalize=True).round(4).to_dict(),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s (events=%d respondents=%d)", out, len(ev), summary["n_respondents"])
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    print(json.dumps(prepare(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
