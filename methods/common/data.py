"""Load the three mode-choice datasets as numeric choice tensors, aligned to
the exact per-seed train/val/test splits used by the OM-LEU Modal runs.

Every method under ``methods/`` consumes :class:`ChoiceDataset`, so all of
them are evaluated on the same events as OM-LEU and the ML baselines in
``<dataset>/results/<run_tag>/seed_<s>/``.

Alignment: a record in ``records.pkl`` carries ``(customer_id, order_date)``;
``<dataset>/data/events.csv`` maps that pair to an ``event_id``; the raw
``.dat`` file is re-filtered exactly as the prepare script does so that
``event_id`` identifies a raw row with numeric level-of-service columns.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUN_TAGS = {
    "swissmetro": "swissmetro_full_beta1_v2",
    "optima": "optima_llama70b",
    "lpmc": "lpmc_llama70b",
}

ALTS = {
    "swissmetro": ["train", "sm", "car"],
    "optima": ["pt", "car", "soft"],
    "lpmc": ["walk", "cycle", "pt", "drive"],
}

# Nests used by the nested-logit method (indices into ALTS[dataset]).
NESTS = {
    "swissmetro": {"rail": [0, 1], "road": [2]},
    "optima": {"car": [1], "non_car": [0, 2]},
    "lpmc": {"active": [0, 1], "motorised": [2, 3]},
}

ALT_FEATURES = {
    "swissmetro": ["time_h", "cost_100chf", "headway_h", "seats"],
    "optima": ["time_h", "cost_chf", "waiting_h", "walking_h", "transfers", "distance_km"],
    "lpmc": ["time_h", "cost_gbp", "access_h", "interchanges", "traffic_pct"],
}
TIME_IDX = 0
COST_IDX = 1

# Optima attitudinal indicators used for the latent "car-loving" attitude
# (the Biogeme hybrid-choice case study uses these seven Likert items).
OPTIMA_INDICATORS = ["Envir01", "Envir02", "Envir03", "Mobil11", "Mobil14", "Mobil16", "Mobil17"]


@dataclass
class ChoiceSplit:
    name: str
    X: np.ndarray            # (N, J, F) alternative attributes, model units
    Z: np.ndarray            # (N, P) person + trip covariates, standardised on train
    y: np.ndarray            # (N,) chosen alternative index
    person: np.ndarray       # (N,) person id (panel key)
    event_id: np.ndarray     # (N,)
    I: Optional[np.ndarray]  # (N, Q) attitudinal indicators (NaN = missing) or None

    @property
    def n(self) -> int:
        return int(self.y.shape[0])


@dataclass
class ChoiceDataset:
    dataset: str
    seed: int
    alts: List[str]
    alt_feature_names: List[str]
    z_names: List[str]
    indicator_names: List[str]
    nests: Dict[str, List[int]]
    splits: Dict[str, ChoiceSplit]
    x_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x_std: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def J(self) -> int:
        return len(self.alts)

    @property
    def F(self) -> int:
        return len(self.alt_feature_names)

    @property
    def P(self) -> int:
        return len(self.z_names)

    def X_std(self, split: str) -> np.ndarray:
        """Alternative attributes standardised per (alt, feature) with train stats."""
        X = self.splits[split].X
        return (X - self.x_mean) / self.x_std


# --------------------------------------------------------------------------- #
# Raw tables (re-filtered exactly like the prepare scripts)
# --------------------------------------------------------------------------- #


def _onehot(s: pd.Series, prefix: str, values=None) -> pd.DataFrame:
    vals = sorted(s.dropna().unique()) if values is None else values
    return pd.DataFrame({f"{prefix}={int(v)}": (s == v).astype(np.float32).values for v in vals}, index=s.index)


def _raw_swissmetro() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, None]:
    df = pd.read_csv(REPO_ROOT / "swissmetro/data/raw/swissmetro.dat", sep=r"\s+")
    df = df[df["CHOICE"].isin([1, 2, 3])].copy()
    df = df[(df["TRAIN_AV"] == 1) & (df["SM_AV"] == 1) & (df["CAR_AV"] == 1)].copy()
    df["person_id"] = "r" + df["ID"].astype(int).astype(str).str.zfill(4)
    df["event_id"] = df["person_id"] + "_s" + (df.groupby("ID").cumcount() + 1).astype(str)
    df["chosen"] = df["CHOICE"].map({1: "train", 2: "sm", 3: "car"})
    ga = (df["GA"] == 1).astype(float).values
    X = np.zeros((len(df), 3, 4), dtype=np.float32)
    X[:, 0] = np.stack([df["TRAIN_TT"] / 60, df["TRAIN_CO"] * (1 - ga) / 100, df["TRAIN_HE"] / 60, np.zeros(len(df))], 1)
    X[:, 1] = np.stack([df["SM_TT"] / 60, df["SM_CO"] * (1 - ga) / 100, df["SM_HE"] / 60, df["SM_SEATS"]], 1)
    X[:, 2] = np.stack([df["CAR_TT"] / 60, df["CAR_CO"] / 100, np.zeros(len(df)), np.zeros(len(df))], 1)
    Z = pd.concat(
        [
            _onehot(df["AGE"], "age", [1, 2, 3, 4, 5, 6]),
            _onehot(df["INCOME"], "income", [0, 1, 2, 3, 4]),
            _onehot(df["PURPOSE"], "purpose", list(range(1, 10))),
            _onehot(df["WHO"], "who", [0, 1, 2, 3]),
            _onehot(df["LUGGAGE"], "luggage", [0, 1, 3]),
            pd.DataFrame({"male": df["MALE"].astype(np.float32), "ga": df["GA"].astype(np.float32),
                          "first_class": df["FIRST"].astype(np.float32)}, index=df.index),
        ],
        axis=1,
    )
    return df, X, Z, None


def _raw_optima() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    df = pd.read_csv(REPO_ROOT / "optima/data/raw/optima.dat", sep=r"\s+")
    df = df[df["Choice"].isin([0, 1, 2])].copy()
    df["person_id"] = "o" + df["ID"].astype(int).astype(str)
    df["event_id"] = df["person_id"] + "_l" + (df.groupby("ID").cumcount() + 1).astype(str)
    df["chosen"] = df["Choice"].map({0: "pt", 1: "car", 2: "soft"})
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).clip(lower=0).astype(float).values
    X = np.zeros((len(df), 3, 6), dtype=np.float32)
    X[:, 0] = np.stack([num("TimePT") / 60, num("MarginalCostPT"), num("WaitingTimePT") / 60,
                        num("WalkingTimePT") / 60, num("NbTransf"), np.zeros(len(df))], 1)
    X[:, 1] = np.stack([num("TimeCar") / 60, num("CostCarCHF"), np.zeros(len(df)), np.zeros(len(df)),
                        np.zeros(len(df)), np.zeros(len(df))], 1)
    X[:, 2] = np.stack([np.zeros(len(df))] * 5 + [num("distance_km")], 1)
    by = df["BirthYear"].where(df["BirthYear"] > 1900)
    age = (2010 - by).fillna((2010 - by).median())
    Z = pd.concat(
        [
            pd.DataFrame({
                "age": age.astype(np.float32),
                "male": (df["Gender"] == 1).astype(np.float32),
                "halffare": (df["HalfFareST"] == 1).astype(np.float32),
                "ga": (df["GenAbST"] == 1).astype(np.float32),
                "line_st": (df["LineRelST"] == 1).astype(np.float32),
                "area_st": (df["AreaRelST"] == 1).astype(np.float32),
                "n_car": df["NbCar"].clip(lower=0).astype(np.float32),
                "n_child": df["NbChild"].clip(lower=0).astype(np.float32),
                "n_household": df["NbHousehold"].clip(lower=0).astype(np.float32),
                "education": df["Education"].clip(lower=0).astype(np.float32),
                "log_distance": np.log1p(df["distance_km"].clip(lower=0)).astype(np.float32),
            }, index=df.index),
            _onehot(df["Income"], "income", [-1, 1, 2, 3, 4, 5, 6]),
            _onehot(df["CarAvail"], "caravail", [-1, 1, 2, 3]),
            _onehot(df["OccupStat"], "occup", [-1] + list(range(1, 10))),
            _onehot(df["FamilSitu"], "family", [-1] + list(range(1, 8))),
            _onehot(df["TypeCommune"], "commune", [1, 2, 4, 5, 6, 7, 8, 9]),
            _onehot(df["TripPurpose"], "trippurpose", [-1, 1, 2, 3]),
            _onehot(df["Region"], "region", list(range(1, 9))),
        ],
        axis=1,
    )
    I = df[OPTIMA_INDICATORS].astype(float).values
    I[(I < 1) | (I > 5)] = np.nan
    return df, X, Z, I.astype(np.float32)


def _raw_lpmc() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, None]:
    df = pd.read_csv(REPO_ROOT / "lpmc/data/raw/lpmc.dat", sep=r"\s+")
    df["person_id"] = "h" + df["household_id"].astype(int).astype(str) + "_p" + df["person_n"].astype(int).astype(str)
    df["event_id"] = "t" + df["trip_id"].astype(int).astype(str)
    df["chosen"] = df["travel_mode"].map({1: "walk", 2: "cycle", 3: "pt", 4: "drive"})
    num = lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float).values
    n = len(df)
    X = np.zeros((n, 4, 5), dtype=np.float32)
    X[:, 0] = np.stack([num("dur_walking"), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)], 1)
    X[:, 1] = np.stack([num("dur_cycling"), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)], 1)
    pt_time = num("dur_pt_access") + num("dur_pt_rail") + num("dur_pt_bus") + num("dur_pt_int")
    X[:, 2] = np.stack([pt_time, num("cost_transit"), num("dur_pt_access"), num("pt_interchanges"), np.zeros(n)], 1)
    X[:, 3] = np.stack([num("dur_driving"), num("cost_driving_fuel") + num("cost_driving_ccharge"),
                        np.zeros(n), np.zeros(n), num("driving_traffic_percent")], 1)
    hour = pd.to_numeric(df["start_time"], errors="coerce").fillna(12.0)
    Z = pd.concat(
        [
            pd.DataFrame({
                "age": df["age"].astype(np.float32),
                "female": df["female"].astype(np.float32),
                "licence": df["driving_license"].astype(np.float32),
                "log_distance": np.log1p(df["distance"].astype(float)).astype(np.float32),
                "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float32),
                "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float32),
                "weekend": df["day_of_week"].isin([6, 7]).astype(np.float32),
                "month_sin": np.sin(2 * np.pi * df["travel_month"] / 12).astype(np.float32),
                "month_cos": np.cos(2 * np.pi * df["travel_month"] / 12).astype(np.float32),
            }, index=df.index),
            _onehot(df["car_ownership"], "cars", [0, 1, 2]),
            _onehot(df["faretype"], "fare", [1, 2, 3, 4, 5]),
            _onehot(df["purpose"], "purpose", [1, 2, 3, 4, 5]),
        ],
        axis=1,
    )
    return df, X, Z, None


_RAW_BUILDERS = {"swissmetro": _raw_swissmetro, "optima": _raw_optima, "lpmc": _raw_lpmc}
_RAW_CACHE: dict = {}


def _raw(dataset: str):
    if dataset not in _RAW_CACHE:
        _RAW_CACHE[dataset] = _RAW_BUILDERS[dataset]()
    return _RAW_CACHE[dataset]


# --------------------------------------------------------------------------- #
# Split alignment
# --------------------------------------------------------------------------- #


def run_tag(dataset: str) -> str:
    return os.environ.get(f"METHODS_RUN_TAG_{dataset.upper()}", DEFAULT_RUN_TAGS[dataset])


def records_path(dataset: str, seed: int) -> Path:
    return REPO_ROOT / dataset / "results" / run_tag(dataset) / f"seed_{seed}" / "omleu" / "records.pkl"


def _event_lookup(dataset: str) -> Dict[tuple, str]:
    ev = pd.read_csv(REPO_ROOT / dataset / "data" / "events.csv")
    id_col = "person_id" if "person_id" in ev.columns else "respondent_id"
    t_col = "timestamp" if "timestamp" in ev.columns else "pseudo_date"
    e_col = "event_id" if "event_id" in ev.columns else "scenario_id"
    ts = pd.to_datetime(ev[t_col])
    out: Dict[tuple, str] = {}
    for pid, t, eid in zip(ev[id_col], ts, ev[e_col]):
        out[(str(pid), t)] = str(eid)
    return out


def load_dataset(dataset: str, seed: int) -> ChoiceDataset:
    """Build a :class:`ChoiceDataset` aligned to ``records.pkl`` of ``seed``."""
    if dataset not in ALTS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(ALTS)}")
    path = records_path(dataset, seed)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the OM-LEU pipeline for {dataset} seed {seed} first")
    recs = pickle.load(open(path, "rb"))
    df, X_all, Z_all, I_all = _raw(dataset)
    row_of = {eid: i for i, eid in enumerate(df["event_id"].values)}
    lookup = _event_lookup(dataset)
    alts = ALTS[dataset]
    alt_idx = {a: i for i, a in enumerate(alts)}

    z_cols = list(Z_all.columns)
    Z_np = Z_all.values.astype(np.float32)
    splits: Dict[str, ChoiceSplit] = {}
    for name in ("train", "val", "test"):
        rows, ys, persons, eids = [], [], [], []
        for r in recs[name]:
            key = (str(r["customer_id"]), pd.Timestamp(r["order_date"]))
            eid = lookup.get(key)
            if eid is None or eid not in row_of:
                raise KeyError(f"{dataset} seed {seed}: record {key} has no raw row")
            i = row_of[eid]
            if df["chosen"].iat[i] != r["chosen_asin"]:
                raise AssertionError(f"{dataset} {eid}: raw chosen {df['chosen'].iat[i]} != record {r['chosen_asin']}")
            rows.append(i)
            ys.append(alt_idx[r["chosen_asin"]])
            persons.append(str(r["customer_id"]))
            eids.append(eid)
        rows = np.asarray(rows)
        splits[name] = ChoiceSplit(
            name=name,
            X=X_all[rows],
            Z=Z_np[rows],
            y=np.asarray(ys, dtype=np.int64),
            person=np.asarray(persons),
            event_id=np.asarray(eids),
            I=None if I_all is None else I_all[rows],
        )
    # Standardise Z on train statistics (keep zero-variance columns at 0).
    mu = splits["train"].Z.mean(0)
    sd = splits["train"].Z.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    for s in splits.values():
        s.Z = ((s.Z - mu) / sd).astype(np.float32)
    x_mean = splits["train"].X.mean(0)
    x_std = splits["train"].X.std(0)
    x_std = np.where(x_std > 1e-6, x_std, 1.0)
    return ChoiceDataset(
        dataset=dataset,
        seed=seed,
        alts=alts,
        alt_feature_names=ALT_FEATURES[dataset],
        z_names=z_cols,
        indicator_names=OPTIMA_INDICATORS if I_all is not None else [],
        nests=NESTS[dataset],
        splits=splits,
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
    )
