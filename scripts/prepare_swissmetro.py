"""Prepare the Swissmetro stated-preference survey for PO-LEU.

Input: ``swissmetro.dat`` (whitespace-separated; Bierlaire, Axhausen & Abay
2001; https://transp-or.epfl.ch/pythonbiogeme/examples_swissmetro.html).
1,192 respondents x 9 scenarios, each choosing among regular train, the
proposed Swissmetro maglev and car for one intercity trip, with per-
scenario travel time, cost and headway per mode.

Outputs (``--out``, default ``<repo>/swissmetro/data/``):

* ``events.csv``      — one row per (respondent, scenario) with the chosen
                        mode, the trip descriptors and the slate
                        (``slate_alt_ids``: the available modes).
* ``persons.csv``     — one row per respondent: age / income classes,
                        gender, first-class habit, GA ticket, and paraphrased
                        ``profile_lines`` for the context string.
* ``impressions.csv`` — per-(scenario, mode) travel time, cost, headway and
                        Swissmetro seating as shown to the respondent.
* ``properties.csv``  — the three modes as catalog rows.
* ``prepare_summary.json``

Conventions
-----------
* Scenarios where any mode is unavailable are dropped (``--availability
  all``, default) so ``J`` is uniform. In the raw file ``CAR_AV = 0`` for
  non-car-owners, so this removes whole respondents (about 16% of rows).
* Following the standard Biogeme specification, rail costs are set to 0 for
  GA season-ticket holders (the raw file stores the annual ticket price
  in ``TRAIN_CO`` for them, and ``SM_CO`` ignores the GA). The context
  string tells the generator the respondent holds a GA.
* ``pseudo_date`` = 1998-01-01 + 7 days x (scenario index - 1). The
  scenarios are not time-ordered; the pseudo-date only feeds the
  per-respondent split (7 train / 1 val / 1 test) and the 30-day
  "recent choices" window.

Usage
-----
    python scripts/prepare_swissmetro.py --raw swissmetro.dat [--out swissmetro/data]
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

logger = logging.getLogger("prepare_swissmetro")

ALT_IDS: tuple[str, ...] = ("train", "sm", "car")
ALT_LABELS: dict[str, str] = {
    "train": "Regular train (Swiss intercity rail)",
    "sm": "Swissmetro (proposed 500 km/h underground maglev train)",
    "car": "Private car",
}
ALT_BRAND: dict[str, str] = {"train": "regular rail", "sm": "maglev rail", "car": "private car"}
CHOICE_TO_ALT: dict[int, str] = {1: "train", 2: "sm", 3: "car"}

PURPOSE_LABEL: dict[int, str] = {
    1: "commuting", 2: "shopping", 3: "business", 4: "leisure",
    5: "commute home from work", 6: "trip home from shopping",
    7: "trip home from a business appointment", 8: "trip home from a leisure activity",
    9: "other",
}
PURPOSE_CATEGORY: dict[int, str] = {
    1: "commute", 2: "shopping", 3: "business", 4: "leisure",
    5: "commute", 6: "shopping", 7: "business", 8: "leisure", 9: "other",
}
PAYER_LABEL: dict[int, str] = {
    0: "", 1: "paying the fare themselves", 2: "with the employer paying",
    3: "splitting the cost with the employer",
}
LUGGAGE_LABEL: dict[int, str] = {
    0: "travelling without luggage", 1: "carrying one piece of luggage",
    3: "carrying several pieces of luggage",
}
AGE_CLASS: dict[int, str] = {
    1: "under_25", 2: "25_to_39", 3: "39_to_54", 4: "54_to_65", 5: "over_65", 6: "unknown",
}
INCOME_CLASS: dict[int, str] = {
    0: "under_50k", 1: "under_50k", 2: "50k_to_100k", 3: "over_100k", 4: "unknown",
}
TICKET_LABEL: dict[int, str] = {
    0: "", 1: "usually buys return tickets with a half-fare card",
    2: "usually buys one-way tickets with a half-fare card",
    3: "usually buys full-price return tickets", 4: "usually buys full-price one-way tickets",
    5: "usually buys half-day tickets", 6: "holds an annual season ticket",
    7: "holds a junior or senior annual season ticket",
    8: "holds an evening travel card", 9: "usually travels on group tickets", 10: "",
}
CANTON: dict[int, str] = {
    1: "Zurich", 2: "Bern", 3: "Lucerne", 4: "Uri", 5: "Schwyz", 6: "Obwalden",
    7: "Nidwalden", 8: "Glarus", 9: "Zug", 10: "Fribourg", 11: "Solothurn",
    12: "Basel-Stadt", 13: "Basel-Landschaft", 14: "Schaffhausen",
    15: "Appenzell Ausserrhoden", 16: "Appenzell Innerrhoden", 17: "St. Gallen",
    18: "Graubunden", 19: "Aargau", 20: "Thurgau", 21: "Ticino", 22: "Vaud",
    23: "Valais", 24: "Neuchatel", 25: "Geneva", 26: "Jura",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True, help="Path to swissmetro.dat")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "swissmetro" / "data")
    p.add_argument("--availability", choices=["all", "any"], default="all",
                   help="'all': keep scenarios where every mode is available (uniform J=3).")
    p.add_argument("--day-spacing", type=int, default=7,
                   help="Days between consecutive pseudo-dated scenarios (default 7).")
    p.add_argument("--log-level", default="INFO")
    return p


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+")
    required = ["ID", "PURPOSE", "FIRST", "TICKET", "WHO", "LUGGAGE", "AGE", "MALE",
                "INCOME", "GA", "ORIGIN", "DEST", "TRAIN_AV", "CAR_AV", "SM_AV",
                "TRAIN_TT", "TRAIN_CO", "TRAIN_HE", "SM_TT", "SM_CO", "SM_HE",
                "SM_SEATS", "CAR_TT", "CAR_CO", "CHOICE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    return df


def prepare(args: argparse.Namespace) -> dict:
    df = load_raw(args.raw)
    n_raw = len(df)
    df = df[df["CHOICE"].isin([1, 2, 3])].copy()
    n_choice = len(df)
    if args.availability == "all":
        mask = (df["TRAIN_AV"] == 1) & (df["SM_AV"] == 1) & (df["CAR_AV"] == 1)
        df = df[mask].copy()
    n_avail = len(df)

    # String ids with a letter prefix so pandas never autotypes them to int
    # on either side of the events/persons join.
    df["respondent_id"] = "r" + df["ID"].astype(int).astype(str).str.zfill(4)
    df["scenario_idx"] = df.groupby("ID").cumcount() + 1
    df["scenario_id"] = df["respondent_id"] + "_s" + df["scenario_idx"].astype(str)
    base = pd.Timestamp("1998-01-01")
    df["pseudo_date"] = [
        (base + pd.to_timedelta(int(args.day_spacing) * (int(i) - 1), unit="D")).strftime("%Y-%m-%d")
        for i in df["scenario_idx"]
    ]
    ga = df["GA"].astype(int) == 1
    # Biogeme convention: rail costs vanish for GA holders.
    df["train_cost"] = np.where(ga, 0.0, df["TRAIN_CO"].astype(float))
    df["sm_cost"] = np.where(ga, 0.0, df["SM_CO"].astype(float))
    df["car_cost"] = df["CAR_CO"].astype(float)
    df["alt_id"] = df["CHOICE"].astype(int).map(CHOICE_TO_ALT)
    cost_by_alt = {"train": df["train_cost"], "sm": df["sm_cost"], "car": df["car_cost"]}
    df["cost_chf"] = np.select(
        [df["alt_id"] == a for a in ALT_IDS], [cost_by_alt[a] for a in ALT_IDS], default=0.0
    )

    # ---- events -----------------------------------------------------------
    ev = pd.DataFrame({
        "respondent_id": df["respondent_id"],
        "scenario_id": df["scenario_id"],
        "scenario_idx": df["scenario_idx"].astype(int),
        "pseudo_date": df["pseudo_date"],
        "alt_id": df["alt_id"],
        "alt_label": df["alt_id"].map(ALT_LABELS),
        "cost_chf": df["cost_chf"].astype(float),
        "purpose_label": df["PURPOSE"].astype(int).map(PURPOSE_LABEL).fillna("other"),
        "purpose_category": df["PURPOSE"].astype(int).map(PURPOSE_CATEGORY).fillna("other"),
        "payer_label": df["WHO"].astype(int).map(PAYER_LABEL).fillna(""),
        "luggage_label": df["LUGGAGE"].astype(int).map(LUGGAGE_LABEL).fillna(""),
        "origin_label": df["ORIGIN"].astype(int).map(CANTON).fillna(""),
        "dest_label": df["DEST"].astype(int).map(CANTON).fillna(""),
        "ga": df["GA"].astype(int),
        "slate_alt_ids": "|".join(ALT_IDS),
    })
    # The adapter's "category" is the coarse purpose (5 buckets).
    ev["purpose_label"], ev["purpose_detail"] = ev["purpose_category"], ev["purpose_label"]

    # ---- impressions ------------------------------------------------------
    imp_rows: list[dict] = []
    for rec in df.to_dict("records"):
        sid = rec["scenario_id"]
        imp_rows.append({"scenario_id": sid, "alt_id": "train",
                         "travel_time_min": float(rec["TRAIN_TT"]), "cost_chf": float(rec["train_cost"]),
                         "headway_min": float(rec["TRAIN_HE"]), "airline_seating": 0})
        imp_rows.append({"scenario_id": sid, "alt_id": "sm",
                         "travel_time_min": float(rec["SM_TT"]), "cost_chf": float(rec["sm_cost"]),
                         "headway_min": float(rec["SM_HE"]), "airline_seating": int(rec["SM_SEATS"])})
        imp_rows.append({"scenario_id": sid, "alt_id": "car",
                         "travel_time_min": float(rec["CAR_TT"]), "cost_chf": float(rec["car_cost"]),
                         "headway_min": 0.0, "airline_seating": 0})
    imp = pd.DataFrame(imp_rows)

    # ---- properties -------------------------------------------------------
    props = pd.DataFrame({
        "prop_id": list(ALT_IDS),
        "title": [ALT_LABELS[a] for a in ALT_IDS],
        "category": ["intercity travel mode"] * 3,
        "price": [float(imp.loc[imp["alt_id"] == a, "cost_chf"].median()) for a in ALT_IDS],
        "brand": [ALT_BRAND[a] for a in ALT_IDS],
    })

    # ---- persons ----------------------------------------------------------
    p_rows: list[dict] = []
    for rid, g in df.groupby("respondent_id", sort=True):
        first = g.iloc[0]
        lines: list[str] = []
        if int(first["GA"]) == 1:
            lines.append("holds a GA annual season ticket covering all Swiss rail travel")
        ticket = TICKET_LABEL.get(int(g["TICKET"].mode().iloc[0]), "")
        if ticket and int(first["GA"]) != 1:
            lines.append(ticket)
        if int(first["FIRST"]) == 1:
            lines.append("usually travels first class")
        grp = int(first.get("GROUP", 0))
        if grp == 2:
            lines.append("currently makes this kind of trip by train")
        elif grp == 3:
            lines.append("currently makes this kind of trip by car")
        p_rows.append({
            "respondent_id": str(rid),
            "age_class": AGE_CLASS.get(int(first["AGE"]), "unknown"),
            "income_class": INCOME_CLASS.get(int(first["INCOME"]), "unknown"),
            "gender_label": "Male" if int(first["MALE"]) == 1 else "Female",
            "first_class": int(first["FIRST"]),
            "ga": int(first["GA"]),
            "survey_group": grp,
            "profile_lines": "|".join(lines),
        })
    persons = pd.DataFrame(p_rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "events.csv", index=False)
    persons.to_csv(out / "persons.csv", index=False)
    imp.to_csv(out / "impressions.csv", index=False)
    props.to_csv(out / "properties.csv", index=False)
    summary = {
        "raw": str(args.raw), "availability": args.availability,
        "n_rows_raw": int(n_raw), "n_rows_with_choice": int(n_choice),
        "n_events": int(n_avail), "n_respondents": int(ev["respondent_id"].nunique()),
        "events_per_respondent": ev.groupby("respondent_id").size().value_counts().to_dict(),
        "choice_shares": ev["alt_id"].value_counts(normalize=True).round(4).to_dict(),
        "ga_share": float(persons["ga"].mean()),
        "purpose_shares": ev["purpose_label"].value_counts(normalize=True).round(4).to_dict(),
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("wrote %s (events=%d respondents=%d)", out, n_avail, summary["n_respondents"])
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    summary = prepare(args)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
