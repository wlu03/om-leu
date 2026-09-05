"""Prepare the Expedia Group RecTour research dataset for PO-LEU.

Turns the raw RecTour release (one row per *search* with a pipe-delimited
``impressions`` field, or the flattened one-row-per-impression variant)
into the four CSVs the ``expedia_rectour`` adapter reads:

* ``events.csv``      — one row per search with a chosen property. Carries
                        the search parameters and the trimmed displayed
                        slate (``slate_prop_ids``, pipe-joined, exactly J).
* ``persons.csv``     — one row per user with history-derived pseudo
                        demographics (party size, kids, price tier ...),
                        computed on each user's earliest 80% of events so
                        nothing from the held-out tail leaks into z_d.
* ``impressions.csv`` — per-(search, property) attributes as displayed
                        (price tier, ratings, flags, rank, ad flag) for
                        every slate member of every kept search.
* ``properties.csv``  — per-property constants (title, star band, medians
                        and optional amenities) for slate members that
                        are never chosen and so have no event row.

Choice definition
-----------------
``--label-mode book`` keeps only searches with a transaction (the booked
property is the choice). ``--label-mode click_or_book`` (default, needed
because RecTour excludes users with more than four bookings, so booking-
only histories are too short for a per-user chronological split) uses the
booked property when there is one and otherwise the most-clicked property
(ties broken by display rank). ``label_kind`` records which one applied.

Slate
-----
The chosen property plus the ``J-1`` other impressions with the best
display rank (``--negatives-policy top``) or a seeded random draw
(``random``). Searches with fewer than J impressions are dropped. The
raw file is never modified.

Usage
-----
    python scripts/prepare_expedia_rectour.py \\
        --raw /path/to/rectour_searches.csv \\
        [--amenities /path/to/property_amenities.csv] \\
        [--out expedia_rectour/data] [--J 10] [--min-events 3] \\
        [--max-users 3000] [--seed 42]
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

logger = logging.getLogger("prepare_expedia_rectour")

# Documented impression field order (Woznica & Krasnodebski, RecTour 2021).
IMPR_FIELDS: tuple[str, ...] = (
    "rank",
    "prop_id",
    "is_travel_ad",
    "review_rating",
    "review_count",
    "star_rating",
    "is_free_cancellation",
    "is_drr",
    "price_bucket",
    "num_clicks",
    "is_trans",
)

SEARCH_COLUMNS: tuple[str, ...] = (
    "user_id",
    "search_id",
    "search_timestamp",
    "point_of_sale",
    "geo_location_country",
    "is_mobile",
    "destination_id",
    "checkin_date",
    "checkout_date",
    "adult_count",
    "child_count",
    "infant_count",
    "room_count",
    "sort_type",
    "applied_filters",
)

AMENITY_LABELS: dict[str, str] = {
    "FreeWiFi": "free WiFi",
    "WiFi": "WiFi",
    "HighSpeedInternet": "high-speed internet",
    "FreeBreakfast": "free breakfast",
    "FreeParking": "free parking",
    "Parking": "parking",
    "SwimmingPool": "a pool",
    "PrivatePool": "a private pool",
    "Gym": "a gym",
    "SpaServices": "spa services",
    "HotTub": "a hot tub",
    "Bar": "a bar",
    "AirConditioning": "air conditioning",
    "AirportTransfer": "airport transfer",
    "FreeAirportTransportation": "free airport shuttle",
    "PetsAllowed": "pets allowed",
    "LaundryFacility": "laundry",
    "WasherDryer": "a washer and dryer",
}
_AMENITY_PRIORITY: tuple[str, ...] = (
    "FreeBreakfast", "FreeWiFi", "FreeParking", "SwimmingPool", "Gym",
    "SpaServices", "AirConditioning", "PetsAllowed", "AirportTransfer",
    "FreeAirportTransportation", "HotTub", "Bar", "PrivatePool", "Parking",
    "WiFi", "HighSpeedInternet", "LaundryFacility", "WasherDryer",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, required=True,
                   help="Raw RecTour CSV (nested `impressions` column or flattened rows).")
    p.add_argument("--amenities", type=Path, default=None,
                   help="Optional property-amenities CSV (prop_id + Boolean columns).")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "expedia_rectour" / "data",
                   help="Output directory (default: <repo>/expedia_rectour).")
    p.add_argument("--J", type=int, default=10, help="Choice-set size (default 10).")
    p.add_argument("--label-mode", choices=["book", "click_or_book"],
                   default="click_or_book")
    p.add_argument("--negatives-policy", choices=["top", "random"], default="top",
                   help="Which J-1 non-chosen impressions form the slate.")
    p.add_argument("--min-events", type=int, default=3,
                   help="Keep users with at least this many labelled searches.")
    p.add_argument("--max-users", type=int, default=3000,
                   help="Random cap on the number of users written (0 = no cap).")
    p.add_argument("--max-raw-rows", type=int, default=None,
                   help="Debug: read only the first N raw rows.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _parse_nested_impressions(raw: pd.DataFrame) -> pd.DataFrame:
    """Explode the pipe/comma `impressions` field into one row per impression."""
    rows: list[dict] = []
    n_bad = 0
    for rec in raw.to_dict("records"):
        s = rec.get("impressions")
        if not isinstance(s, str) or not s.strip():
            continue
        base = {c: rec.get(c) for c in SEARCH_COLUMNS if c in rec}
        for item in s.split("|"):
            parts = [x.strip() for x in item.split(",")]
            if len(parts) != len(IMPR_FIELDS):
                n_bad += 1
                continue
            row = dict(base)
            row.update(dict(zip(IMPR_FIELDS, parts)))
            rows.append(row)
    if n_bad:
        logger.warning("skipped %d malformed impression items", n_bad)
    return pd.DataFrame(rows)


def load_long_impressions(raw_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Return a long frame (one row per impression) whatever the raw layout."""
    raw = pd.read_csv(raw_path, nrows=max_rows, dtype=str, keep_default_na=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    if "impressions" in raw.columns:
        logger.info("raw layout: nested impressions (%d searches)", len(raw))
        long = _parse_nested_impressions(raw)
    elif "prop_id" in raw.columns:
        logger.info("raw layout: flattened impressions (%d rows)", len(raw))
        long = raw
    else:
        raise SystemExit(
            f"{raw_path}: expected either an `impressions` column (nested "
            f"RecTour layout) or `prop_id` columns (flattened layout); got "
            f"{list(raw.columns)[:20]}"
        )
    required = ["user_id", "search_id", "search_timestamp", "prop_id"]
    missing = [c for c in required if c not in long.columns]
    if missing:
        raise SystemExit(f"raw data is missing required columns: {missing}")
    for c in ("rank", "review_rating", "review_count", "star_rating",
              "price_bucket", "num_clicks", "adult_count", "child_count",
              "infant_count", "room_count"):
        if c in long.columns:
            long[c] = pd.to_numeric(long[c], errors="coerce")
    for c in ("is_travel_ad", "is_free_cancellation", "is_drr", "is_trans",
              "is_mobile"):
        if c in long.columns:
            long[c] = long[c].map(_to_flag).astype(int)
        else:
            long[c] = 0
    if "rank" not in long.columns:
        long["rank"] = long.groupby("search_id").cumcount() + 1
    long["user_id"] = long["user_id"].astype(str)
    long["search_id"] = long["search_id"].astype(str)
    long["prop_id"] = long["prop_id"].astype(str)
    return long


def _to_flag(v) -> int:
    if isinstance(v, str):
        return 1 if v.strip().lower() in {"1", "true", "t", "yes", "y", "1.0"} else 0
    try:
        return 1 if float(v) >= 0.5 else 0
    except (TypeError, ValueError):
        return 0


# --------------------------------------------------------------------------- #
# Choice + slate construction
# --------------------------------------------------------------------------- #


def choose_and_slate(
    long: pd.DataFrame, *, J: int, label_mode: str, negatives_policy: str, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (events, impressions-of-kept-slates, stats)."""
    rng = np.random.default_rng(seed)
    long = long.sort_values(["search_id", "rank"], kind="mergesort")
    events: list[dict] = []
    kept_imps: list[pd.DataFrame] = []
    stats = {"n_searches": 0, "no_label": 0, "short_slate": 0,
             "booked": 0, "clicked": 0}
    for sid, g in long.groupby("search_id", sort=False):
        stats["n_searches"] += 1
        g = g.drop_duplicates(subset=["prop_id"], keep="first")
        booked = g[g["is_trans"] == 1]
        chosen = None
        label_kind = None
        if len(booked) > 0:
            chosen = booked.sort_values("rank").iloc[0]
            label_kind = "book"
        elif label_mode == "click_or_book" and "num_clicks" in g.columns:
            clicked = g[g["num_clicks"].fillna(0) > 0]
            if len(clicked) > 0:
                chosen = clicked.sort_values(["num_clicks", "rank"],
                                             ascending=[False, True]).iloc[0]
                label_kind = "click"
        if chosen is None:
            stats["no_label"] += 1
            continue
        if len(g) < J:
            stats["short_slate"] += 1
            continue
        others = g[g["prop_id"] != chosen["prop_id"]]
        if negatives_policy == "top":
            negs = others.sort_values("rank").head(J - 1)
        else:
            idx = rng.choice(len(others), size=J - 1, replace=False)
            negs = others.iloc[np.sort(idx)]
        slate = pd.concat([chosen.to_frame().T, negs], axis=0)
        slate_ids = slate["prop_id"].astype(str).tolist()
        stats["booked" if label_kind == "book" else "clicked"] += 1
        ev = {c: chosen.get(c) for c in SEARCH_COLUMNS if c in chosen.index}
        ev.update({
            "search_id": str(sid),
            "prop_id": str(chosen["prop_id"]),
            "label_kind": label_kind,
            "chosen_star_rating": chosen.get("star_rating"),
            "chosen_price_bucket": chosen.get("price_bucket"),
            "n_impressions": int(len(g)),
            "slate_prop_ids": "|".join(slate_ids),
        })
        events.append(ev)
        kept_imps.append(slate[[c for c in ("search_id", "prop_id", "rank",
                                            "is_travel_ad", "review_rating",
                                            "review_count", "star_rating",
                                            "is_free_cancellation", "is_drr",
                                            "price_bucket") if c in slate.columns]])
    ev_df = pd.DataFrame(events)
    imp_df = pd.concat(kept_imps, axis=0, ignore_index=True) if kept_imps else pd.DataFrame()
    return ev_df, imp_df, stats


def _derive_search_fields(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["search_timestamp"] = pd.to_datetime(ev["search_timestamp"], errors="coerce")
    ev = ev.dropna(subset=["search_timestamp"])
    for c in ("checkin_date", "checkout_date"):
        if c in ev.columns:
            ev[c] = pd.to_datetime(ev[c], errors="coerce")
    if "checkin_date" in ev.columns and "checkout_date" in ev.columns:
        ev["length_of_stay"] = (ev["checkout_date"] - ev["checkin_date"]).dt.days
        ev["booking_window"] = (
            ev["checkin_date"].dt.normalize() - ev["search_timestamp"].dt.normalize()
        ).dt.days
    else:
        ev["length_of_stay"] = np.nan
        ev["booking_window"] = np.nan
    for c in ("adult_count", "child_count", "infant_count", "room_count"):
        if c not in ev.columns:
            ev[c] = np.nan
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev["price_bucket"] = pd.to_numeric(ev["chosen_price_bucket"], errors="coerce")
    # price must be non-negative float for the invariants; missing tier ->
    # median of observed tiers (chosen side only; slate members use the
    # impressions file).
    med = float(ev["price_bucket"].median()) if ev["price_bucket"].notna().any() else 3.0
    ev["price_bucket"] = ev["price_bucket"].fillna(med).clip(lower=1.0, upper=5.0)
    ev["destination_label"] = "destination " + ev["destination_id"].astype(str).str.strip()
    ev.loc[ev["destination_id"].isna(), "destination_label"] = "Unknown"
    return ev


# --------------------------------------------------------------------------- #
# Persons + properties
# --------------------------------------------------------------------------- #


def _party_bucket(x: float) -> str:
    # Word labels (not "1".."5+") so pandas never autotypes the column to
    # int and the YAML categorical_to_int keys match exactly.
    if not np.isfinite(x) or x < 1.5:
        return "solo"
    if x < 2.5:
        return "pair"
    if x < 3.5:
        return "three"
    if x < 4.5:
        return "four"
    return "five_plus"


def build_persons(ev: pd.DataFrame, *, history_frac: float = 0.8) -> pd.DataFrame:
    """Per-user pseudo-demographics from the earliest ``history_frac`` events."""
    rows: list[dict] = []
    for uid, g in ev.sort_values("search_timestamp").groupby("user_id", sort=False):
        n = len(g)
        n_hist = max(1, int(np.ceil(history_frac * n)))
        h = g.iloc[:n_hist]
        party = (h["adult_count"].fillna(0) + h["child_count"].fillna(0))
        party = party[party >= 1]
        party_mean = float(party.mean()) if len(party) else 1.0
        kids = int((h["child_count"].fillna(0) > 0).any())
        tier = float(h["price_bucket"].mean()) if h["price_bucket"].notna().any() else 3.0
        tier_bucket = "low" if tier < 2.5 else ("high" if tier > 3.5 else "mid")
        mobile = float(pd.to_numeric(h["is_mobile"], errors="coerce").fillna(0).mean()) if "is_mobile" in h.columns else 0.0
        rows.append({
            "user_id": str(uid),
            "n_searches_hist": int(n_hist),
            "party_size_bucket": _party_bucket(party_mean),
            "has_kids": kids,
            "typical_price_tier_bucket": tier_bucket,
            "mobile_share": round(mobile, 3),
            "dominant_country": (
                str(h["geo_location_country"].mode().iloc[0])
                if "geo_location_country" in h.columns and h["geo_location_country"].notna().any()
                else ""
            ),
            "dominant_point_of_sale": (
                str(h["point_of_sale"].mode().iloc[0])
                if "point_of_sale" in h.columns and h["point_of_sale"].notna().any()
                else ""
            ),
        })
    return pd.DataFrame(rows)


def _amenity_phrase(rec: dict) -> str:
    present = [
        AMENITY_LABELS[k] for k in _AMENITY_PRIORITY
        if k in rec and _to_flag(rec[k]) == 1
    ]
    # Collapse redundant pairs (FreeWiFi ⊃ WiFi, FreeParking ⊃ Parking).
    if "free WiFi" in present and "WiFi" in present:
        present.remove("WiFi")
    if "free parking" in present and "parking" in present:
        present.remove("parking")
    return ", ".join(present[:6])


def build_properties(
    imp: pd.DataFrame, ev: pd.DataFrame, amenities_path: Path | None,
) -> pd.DataFrame:
    """Per-property constants for every slate member."""
    g = imp.groupby("prop_id")
    props = pd.DataFrame({
        "prop_id": g.size().index.astype(str),
        "n_impressions": g.size().to_numpy(),
        "star_rating": g["star_rating"].median().to_numpy() if "star_rating" in imp.columns else np.nan,
        "review_rating": g["review_rating"].median().to_numpy() if "review_rating" in imp.columns else np.nan,
        "review_count": g["review_count"].median().to_numpy() if "review_count" in imp.columns else np.nan,
        "price": g["price_bucket"].median().to_numpy() if "price_bucket" in imp.columns else np.nan,
    })
    # Modal destination per property (from the searches it was shown in).
    dest_map = (
        ev[["search_id", "destination_label"]].drop_duplicates("search_id")
        .set_index("search_id")["destination_label"]
    )
    imp_dest = imp["search_id"].map(dest_map)
    dest_mode = (
        pd.DataFrame({"prop_id": imp["prop_id"], "dest": imp_dest})
        .dropna().groupby("prop_id")["dest"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown")
    )
    props["category"] = props["prop_id"].map(dest_mode).fillna("Unknown")
    amen: dict[str, str] = {}
    if amenities_path is not None:
        am = pd.read_csv(amenities_path, dtype={"prop_id": str})
        am.columns = [str(c).strip() for c in am.columns]
        for rec in am.to_dict("records"):
            phrase = _amenity_phrase(rec)
            if phrase:
                amen[str(rec["prop_id"])] = phrase

    def _title(r) -> str:
        t = f"Property #{r['prop_id']}"
        a = amen.get(str(r["prop_id"]))
        if a:
            t += f" with {a}"
        return t

    props["title"] = props.apply(_title, axis=1)
    from src.data.expedia_slates import render_star_band
    props["brand"] = props["star_rating"].map(
        lambda s: render_star_band(float(s)) if pd.notna(s) else "unrated"
    )
    props["price"] = pd.to_numeric(props["price"], errors="coerce").fillna(3.0)
    return props


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def prepare(args: argparse.Namespace) -> dict:
    long = load_long_impressions(args.raw, max_rows=args.max_raw_rows)
    logger.info("long impressions: %d rows, %d searches, %d users",
                len(long), long["search_id"].nunique(), long["user_id"].nunique())

    ev, imp, stats = choose_and_slate(
        long, J=int(args.J), label_mode=args.label_mode,
        negatives_policy=args.negatives_policy, seed=int(args.seed),
    )
    logger.info("choice+slate: %s", json.dumps(stats))
    if ev.empty:
        raise SystemExit("no labelled searches survived; check --label-mode / --J")

    ev = _derive_search_fields(ev)
    # One event per (user, timestamp): duplicate searches (page refresh)
    # keep the first.
    ev = ev.sort_values(["user_id", "search_timestamp", "search_id"]).drop_duplicates(
        subset=["user_id", "search_timestamp"], keep="first"
    )

    counts = ev.groupby("user_id").size()
    keep_users = counts[counts >= int(args.min_events)].index
    ev = ev[ev["user_id"].isin(keep_users)]
    logger.info("users with >= %d events: %d (events %d)",
                int(args.min_events), len(keep_users), len(ev))
    if args.max_users and ev["user_id"].nunique() > int(args.max_users):
        rng = np.random.default_rng(int(args.seed))
        sample = rng.choice(np.sort(ev["user_id"].unique()), size=int(args.max_users),
                            replace=False)
        ev = ev[ev["user_id"].isin(set(sample.tolist()))]
        logger.info("capped to %d users (%d events)", int(args.max_users), len(ev))
    if ev.empty:
        raise SystemExit("no users survived the min-events filter")

    imp = imp[imp["search_id"].isin(set(ev["search_id"]))].copy()
    imp["search_id"] = imp["search_id"].astype(str)
    imp["prop_id"] = imp["prop_id"].astype(str)

    persons = build_persons(ev)
    props = build_properties(imp, ev, args.amenities)
    prop_title = props.set_index("prop_id")["title"]
    ev["prop_label"] = ev["prop_id"].map(prop_title).fillna(
        "Property #" + ev["prop_id"].astype(str)
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ev_out = ev.copy()
    ev_out["search_timestamp"] = ev_out["search_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in ("checkin_date", "checkout_date"):
        if c in ev_out.columns:
            ev_out[c] = pd.to_datetime(ev_out[c]).dt.strftime("%Y-%m-%d")
    cols = [
        "user_id", "search_id", "search_timestamp", "prop_id", "prop_label",
        "destination_id", "destination_label", "price_bucket", "label_kind",
        "checkin_date", "checkout_date", "length_of_stay", "booking_window",
        "adult_count", "child_count", "infant_count", "room_count", "is_mobile",
        "point_of_sale", "geo_location_country", "sort_type", "chosen_star_rating",
        "n_impressions", "slate_prop_ids",
    ]
    cols = [c for c in cols if c in ev_out.columns]
    ev_out[cols].to_csv(out / "events.csv", index=False)
    persons.to_csv(out / "persons.csv", index=False)
    imp.to_csv(out / "impressions.csv", index=False)
    props.to_csv(out / "properties.csv", index=False)

    summary = {
        "raw": str(args.raw),
        "J": int(args.J),
        "label_mode": args.label_mode,
        "negatives_policy": args.negatives_policy,
        "min_events": int(args.min_events),
        "max_users": int(args.max_users),
        "seed": int(args.seed),
        "stats": stats,
        "n_events": int(len(ev)),
        "n_users": int(ev["user_id"].nunique()),
        "n_properties": int(len(props)),
        "n_impressions": int(len(imp)),
        "label_kind_counts": ev["label_kind"].value_counts().to_dict(),
        "events_per_user": {
            "mean": float(ev.groupby("user_id").size().mean()),
            "median": float(ev.groupby("user_id").size().median()),
            "max": int(ev.groupby("user_id").size().max()),
        },
        "date_range": [
            str(ev["search_timestamp"].min()), str(ev["search_timestamp"].max()),
        ],
    }
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s (events=%d users=%d properties=%d impressions=%d)",
                out, len(ev), summary["n_users"], len(props), len(imp))
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    summary = prepare(args)
    print(json.dumps({k: summary[k] for k in ("n_events", "n_users", "n_properties",
                                              "label_kind_counts", "events_per_user")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
