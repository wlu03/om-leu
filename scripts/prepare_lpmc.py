"""Prepare the London Passenger Mode Choice (LPMC) dataset for PO-LEU.

Input: ``lpmc.dat`` (whitespace-separated; Hillel et al. 2018, CS_LPMC).
81,086 trips by 31,954 people in 17,616 households from the London Travel
Demand Survey, April 2012 to March 2015, each with journey-planner
level-of-service attributes for all four modes: walking, cycling, public
transport and driving. ``travel_mode``: 1 walk, 2 cycle, 3 pt, 4 drive.

Outputs (``--out``, default ``<repo>/lpmc/data/``): the generic
mode-choice layout consumed by ``src/data/modechoice_slates.py``
(``events.csv``, ``persons.csv``, ``impressions.csv``, ``properties.csv``).

Conventions
-----------
* Decision-maker = person (``household_id`` + ``person_n``); event = trip,
  timestamped from ``travel_year/month/date`` and ``start_time`` so the
  paper's per-person chronological split applies. Most people have 2-3
  trips; ``--min-events-per-customer 5`` at run time keeps the 2,444
  people with at least five (about 14k trips).
* All four modes are treated as available; licence, car ownership and
  fare type are rendered into the context string.
* Prices in GBP: PT ``cost_transit``; driving ``cost_driving_fuel +
  cost_driving_ccharge``; walking and cycling 0.
* Person attributes (age, gender, licence, car ownership, fare type) are
  static per person in the survey.

Usage
-----
    python scripts/prepare_lpmc.py --raw lpmc/data/raw/lpmc.dat [--out lpmc/data]
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

logger = logging.getLogger("prepare_lpmc")

ALT_IDS: tuple[str, ...] = ("walk", "cycle", "pt", "drive")
MODE_TO_ALT: dict[int, str] = {1: "walk", 2: "cycle", 3: "pt", 4: "drive"}
ALT_LABELS: dict[str, str] = {
    "walk": "Walking", "cycle": "Cycling",
    "pt": "Public transport (bus, tube, rail)", "drive": "Driving",
}
ALT_BRAND: dict[str, str] = {"walk": "active travel", "cycle": "active travel", "pt": "public transport", "drive": "private vehicle"}
# NB: the context-string guard rejects the literal word "education" (a raw
# z_d column name), so the schooling purpose is phrased without it.
PURPOSE_DETAIL: dict[int, str] = {
    1: "commute to or from work", 2: "trip to or from school or college", 3: "trip from or to home",
    4: "trip on the employer's business", 5: "trip not starting or ending at home",
}
PURPOSE_CATEGORY: dict[int, str] = {1: "commute", 2: "schooling", 3: "home-based other", 4: "business", 5: "other"}
FARETYPE: dict[int, str] = {
    1: "pays full adult fares on public transport", 2: "travels on a 16 plus fare",
    3: "travels on a child fare", 4: "travels on a disabled persons fare", 5: "travels free on public transport",
}
CAR_OWN: dict[int, str] = {
    0: "lives in a household without a car", 1: "lives in a household with fewer cars than adults",
    2: "lives in a household with at least one car per adult",
}
WEEKDAY: dict[int, str] = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}


def _fmt_hours(h: float) -> str:
    if not np.isfinite(h) or h < 0:
        return ""
    m = int(round(h * 60))
    if m < 60:
        return f"{m} minutes"
    hh, r = divmod(m, 60)
    return f"{hh} hour{'s' if hh != 1 else ''}" if r == 0 else f"{hh} h {r:02d} min"


def _age_bucket(age: float) -> str:
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


def _daypart(t: float) -> str:
    if t < 6:
        return "in the early morning"
    if t < 10:
        return "in the morning peak"
    if t < 16:
        return "around midday"
    if t < 19:
        return "in the evening peak"
    return "in the evening"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "lpmc" / "data")
    p.add_argument("--min-trips", type=int, default=1,
                   help="Keep persons with at least this many trips (default 1; filter at run time).")
    p.add_argument("--log-level", default="INFO")
    return p


def prepare(args: argparse.Namespace) -> dict:
    df = pd.read_csv(args.raw, sep=r"\s+")
    n_raw = len(df)
    df["person_id"] = "h" + df["household_id"].astype(int).astype(str) + "_p" + df["person_n"].astype(int).astype(str)
    df["event_id"] = "t" + df["trip_id"].astype(int).astype(str)
    hours = pd.to_numeric(df["start_time"], errors="coerce").fillna(12.0).clip(0, 23.99)
    df["timestamp"] = pd.to_datetime(dict(year=df["travel_year"], month=df["travel_month"], day=df["travel_date"]),
                                     errors="coerce") + pd.to_timedelta((hours * 60).round().astype(int), unit="m")
    df = df.dropna(subset=["timestamp"]).copy()
    counts = df.groupby("person_id").size()
    keep = counts[counts >= int(args.min_trips)].index
    df = df[df["person_id"].isin(keep)].copy()
    df["alt_id"] = df["travel_mode"].astype(int).map(MODE_TO_ALT)
    drive_cost = (pd.to_numeric(df["cost_driving_fuel"], errors="coerce").fillna(0)
                  + pd.to_numeric(df["cost_driving_ccharge"], errors="coerce").fillna(0))
    pt_cost = pd.to_numeric(df["cost_transit"], errors="coerce").fillna(0)
    cost = np.select([df["alt_id"] == "pt", df["alt_id"] == "drive"], [pt_cost, drive_cost], default=0.0)

    def _phrase(r) -> str:
        bits = [f"a {PURPOSE_DETAIL.get(int(r['purpose']), 'trip')}"]
        ts = r["timestamp"]
        bits.append(f"on a {WEEKDAY.get(int(r['day_of_week']), 'weekday')} {_daypart(float(r['start_time']))}")
        bits.append(f"in {ts.strftime('%B %Y')}")
        d = float(r["distance"]) / 1000.0
        bits.append(f"about {d:.1f} km as the crow flies")
        return ", ".join(bits)

    ev = pd.DataFrame({
        "person_id": df["person_id"],
        "event_id": df["event_id"],
        "timestamp": df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "alt_id": df["alt_id"],
        "alt_label": df["alt_id"].map(ALT_LABELS),
        "cost_gbp": cost.astype(float),
        "purpose_label": df["purpose"].astype(int).map(PURPOSE_CATEGORY).fillna("other"),
        "trip_phrase": [_phrase(r) for r in df.to_dict("records")],
        "slate_alt_ids": "|".join(ALT_IDS),
    })

    imp_rows: list[dict] = []
    for r in df.to_dict("records"):
        eid = r["event_id"]
        imp_rows.append({"event_id": eid, "alt_id": "walk", "price": 0.0,
                         "travel_time": _fmt_hours(float(r["dur_walking"])), "journey_details": "", "traffic": ""})
        imp_rows.append({"event_id": eid, "alt_id": "cycle", "price": 0.0,
                         "travel_time": _fmt_hours(float(r["dur_cycling"])), "journey_details": "", "traffic": ""})
        acc, rail, bus, intc = (float(r["dur_pt_access"]), float(r["dur_pt_rail"]), float(r["dur_pt_bus"]), float(r["dur_pt_int"]))
        n_int = int(r["pt_interchanges"])
        details = [f"{int(round(acc * 60))} min walking to and from stops"]
        if rail > 0:
            details.append(f"{int(round(rail * 60))} min on rail or tube")
        if bus > 0:
            details.append(f"{int(round(bus * 60))} min on the bus")
        details.append("no interchange" if n_int == 0 else f"{n_int} interchange{'s' if n_int > 1 else ''} ({int(round(intc * 60))} min)")
        imp_rows.append({"event_id": eid, "alt_id": "pt", "price": round(float(r["cost_transit"]), 2),
                         "travel_time": _fmt_hours(acc + rail + bus + intc), "journey_details": "; ".join(details), "traffic": ""})
        cc = float(r["cost_driving_ccharge"])
        dr_details = f"fuel GBP {float(r['cost_driving_fuel']):.2f}" + (f" plus congestion charge GBP {cc:.2f}" if cc > 0 else ", no congestion charge")
        imp_rows.append({"event_id": eid, "alt_id": "drive", "price": round(float(r["cost_driving_fuel"]) + cc, 2),
                         "travel_time": _fmt_hours(float(r["dur_driving"])), "journey_details": dr_details,
                         "traffic": f"predicted traffic variability {100 * float(r['driving_traffic_percent']):.0f}%"})
    imp = pd.DataFrame(imp_rows)

    props = pd.DataFrame({
        "prop_id": list(ALT_IDS),
        "title": [ALT_LABELS[a] for a in ALT_IDS],
        "category": ["travel mode"] * 4,
        "price": [float(imp.loc[imp["alt_id"] == a, "price"].median()) for a in ALT_IDS],
        "brand": [ALT_BRAND[a] for a in ALT_IDS],
    })

    p_rows: list[dict] = []
    for pid, g in df.groupby("person_id", sort=True):
        f = g.iloc[0]
        lines: list[str] = []
        lines.append("holds a driving licence" if int(f["driving_license"]) == 1 else "does not hold a driving licence")
        co = CAR_OWN.get(int(f["car_ownership"]), "")
        if co:
            lines.append(co)
        ft = FARETYPE.get(int(f["faretype"]), "")
        if ft:
            lines.append(ft)
        age = float(f["age"])
        p_rows.append({
            "person_id": pid,
            "age_bucket": _age_bucket(age),
            "gender_label": "Female" if int(f["female"]) == 1 else "Male",
            "has_licence": int(f["driving_license"]),
            "car_ownership": int(f["car_ownership"]),
            "profile_lines": "|".join(lines[:4]),
        })
    persons = pd.DataFrame(p_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "events.csv", index=False)
    persons.to_csv(out / "persons.csv", index=False)
    imp.to_csv(out / "impressions.csv", index=False)
    props.to_csv(out / "properties.csv", index=False)
    tp = ev.groupby("person_id").size()
    summary = {
        "raw": str(args.raw), "n_rows_raw": int(n_raw), "n_events": int(len(ev)),
        "n_persons": int(ev["person_id"].nunique()),
        "persons_with_ge5_trips": int((tp >= 5).sum()), "trips_from_persons_ge5": int(tp[tp >= 5].sum()),
        "mode_shares": ev["alt_id"].value_counts(normalize=True).round(4).to_dict(),
        "purpose_shares": ev["purpose_label"].value_counts(normalize=True).round(4).to_dict(),
        "date_range": [str(ev["timestamp"].min()), str(ev["timestamp"].max())],
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s (events=%d persons=%d)", out, len(ev), summary["n_persons"])
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    print(json.dumps(prepare(args), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
