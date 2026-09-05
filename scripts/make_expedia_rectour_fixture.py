"""Generate a synthetic raw file in the Expedia RecTour layout.

Used ONLY to exercise the ``expedia_rectour`` pipeline end to end (prepare
-> adapter -> choice sets -> generation -> training -> leaderboard) before
the real, request-only RecTour release is available. Numbers produced on
this fixture are structural checks, never evidence.

The file mirrors the documented schema (Woznica & Krasnodebski, RecTour
2021): one row per search with a ``|``-delimited ``impressions`` column
whose items are ``,``-delimited ``rank,prop_id,is_travel_ad,review_rating,
review_count,star_rating,is_free_cancellation,is_drr,price_bucket,
num_clicks,is_trans``. A companion amenities CSV is written too.

Choices follow a latent per-user taste over (price tier, star, guest
rating, free cancellation) so a learnable signal exists.

    python scripts/make_expedia_rectour_fixture.py --out expedia_rectour/data/raw \\
        --n-users 200 --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AMENITY_COLS = [
    "AirConditioning", "AirportTransfer", "Bar", "FreeAirportTransportation",
    "FreeBreakfast", "FreeParking", "FreeWiFi", "Gym", "HighSpeedInternet",
    "HotTub", "LaundryFacility", "Parking", "PetsAllowed", "PrivatePool",
    "SpaServices", "SwimmingPool", "WasherDryer", "WiFi",
]


def make_fixture(
    out_dir: Path, *, n_users: int = 200, n_props: int = 3000, n_dest: int = 40,
    searches_per_user: tuple[int, int] = (3, 12), impressions_per_search: tuple[int, int] = (12, 30),
    seed: int = 0,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Property catalogue.
    prop_ids = np.arange(100000, 100000 + n_props)
    prop_dest = rng.integers(1, n_dest + 1, size=n_props)
    prop_star = rng.choice([2.0, 3.0, 3.5, 4.0, 4.5, 5.0], size=n_props,
                           p=[0.1, 0.25, 0.2, 0.25, 0.12, 0.08])
    prop_rating = np.clip(rng.normal(3.9 + 0.2 * (prop_star - 3.5), 0.4, size=n_props), 1.0, 5.0)
    prop_reviews = (rng.lognormal(5.0, 1.0, size=n_props) // 25 * 25).astype(int)
    prop_amen = rng.random((n_props, len(AMENITY_COLS))) < 0.35
    base_tier = np.clip(np.round(prop_star + rng.normal(0, 0.8, size=n_props) - 0.5), 1, 5)

    # User tastes: weights over [price_tier, star, rating, free_cancel, log_reviews].
    taste = rng.normal(0, 1, size=(n_users, 5))
    taste[:, 0] = -np.abs(taste[:, 0]) * 1.2 + rng.normal(0, 0.4, n_users)  # mostly price-averse

    start = pd.Timestamp("2021-06-01")
    rows: list[dict] = []
    sid = 0
    for u in range(n_users):
        uid = f"u{u:06d}"
        n_s = int(rng.integers(searches_per_user[0], searches_per_user[1] + 1))
        home_dest = int(rng.integers(1, n_dest + 1))
        adults = int(rng.choice([1, 2, 2, 2, 3, 4]))
        kids = int(rng.choice([0, 0, 0, 1, 2]))
        pos = int(rng.integers(1, 12))
        country = int(rng.integers(1, 60))
        mobile_p = rng.random()
        times = np.sort(rng.integers(0, 60 * 24 * 60, size=n_s))
        for k in range(n_s):
            sid += 1
            ts = start + pd.Timedelta(minutes=int(times[k]))
            dest = home_dest if rng.random() < 0.6 else int(rng.integers(1, n_dest + 1))
            cand = np.where(prop_dest == dest)[0]
            n_imp = int(rng.integers(impressions_per_search[0], impressions_per_search[1] + 1))
            if len(cand) < n_imp:
                cand = np.concatenate([cand, rng.choice(n_props, size=n_imp - len(cand), replace=False)])
            shown = rng.choice(cand, size=n_imp, replace=False)
            tier = np.clip(base_tier[shown] + rng.integers(-1, 2, size=n_imp), 1, 5)
            free_cancel = (rng.random(n_imp) < 0.55).astype(int)
            drr = (rng.random(n_imp) < 0.2).astype(int)
            ad = (rng.random(n_imp) < 0.08).astype(int)
            feats = np.stack([
                tier, prop_star[shown], prop_rating[shown], free_cancel,
                np.log1p(prop_reviews[shown]) / 3.0,
            ], axis=1)
            util = feats @ taste[u] - 0.08 * np.arange(n_imp) + rng.gumbel(size=n_imp)
            order = np.argsort(-util)
            n_clicks = np.zeros(n_imp, dtype=int)
            n_clicks[order[0]] = int(rng.integers(1, 4))
            if rng.random() < 0.5:
                n_clicks[order[1]] = 1
            is_trans = np.zeros(n_imp, dtype=int)
            if rng.random() < 0.35:
                is_trans[order[0]] = 1
            lead = int(rng.integers(0, 60))
            checkin = (ts + pd.Timedelta(days=lead)).normalize()
            nights = int(rng.integers(1, 7))
            items = []
            for r in range(n_imp):
                j = shown[r]
                items.append(",".join([
                    str(r + 1), str(prop_ids[j]), str(ad[r]),
                    f"{round(prop_rating[j] * 2) / 2:.1f}", str(prop_reviews[j]),
                    f"{prop_star[j]:.1f}", str(free_cancel[r]), str(drr[r]),
                    str(int(tier[r])), str(n_clicks[r]), str(is_trans[r]),
                ]))
            rows.append({
                "user_id": uid,
                "search_id": f"s{sid:08d}",
                "search_timestamp": ts.strftime("%Y-%m-%d %H:%M:00"),
                "point_of_sale": pos,
                "geo_location_country": country,
                "is_mobile": int(rng.random() < mobile_p),
                "destination_id": dest,
                "checkin_date": checkin.strftime("%Y-%m-%d"),
                "checkout_date": (checkin + pd.Timedelta(days=nights)).strftime("%Y-%m-%d"),
                "adult_count": adults,
                "child_count": kids,
                "infant_count": 0,
                "room_count": 1,
                "sort_type": rng.choice(["RECOMMENDED", "PRICE_LOW_TO_HIGH", "GUEST_RATING"],
                                        p=[0.8, 0.12, 0.08]),
                "applied_filters": "" if rng.random() < 0.7 else "STAR:4.0|LODGING:HOTEL",
                "impressions": "|".join(items),
            })
    raw_path = out_dir / "rectour_searches.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)
    amen = pd.DataFrame(prop_amen.astype(int), columns=AMENITY_COLS)
    amen.insert(0, "prop_id", prop_ids)
    amen_path = out_dir / "property_amenities.csv"
    amen.to_csv(amen_path, index=False)
    return raw_path, amen_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-users", type=int, default=200)
    p.add_argument("--n-props", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    raw, amen = make_fixture(a.out, n_users=a.n_users, n_props=a.n_props, seed=a.seed)
    print(f"wrote {raw}\nwrote {amen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
