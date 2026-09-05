"""Plain multiclass LightGBM on the full feature vector (all alternatives'
level of service flattened + person covariates) — the strongest "black box"
ML classifier in Hillel et al.'s systematic review, given the same
information the choice models receive. Not a RUM: any feature can enter
any class score.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from methods.common.data import ChoiceDataset


def _flat(ds: ChoiceDataset, split: str) -> np.ndarray:
    s = ds.splits[split]
    return np.concatenate([s.X.reshape(s.n, -1), s.Z], axis=1).astype(np.float32)


def run(ds: ChoiceDataset, seed: int, **kw) -> dict:
    t0 = time.time()
    tr, va, te = ds.splits["train"], ds.splits["val"], ds.splits["test"]
    params = {"objective": "multiclass", "num_class": ds.J, "learning_rate": 0.05, "num_leaves": 15,
              "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
              "lambda_l2": 1.0, "verbose": -1, "seed": seed, "num_threads": 4}
    dtr = lgb.Dataset(_flat(ds, "train"), label=tr.y)
    dva = lgb.Dataset(_flat(ds, "val"), label=va.y, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    probs = booster.predict(_flat(ds, "test"), num_iteration=booster.best_iteration)
    return {"probs_test": probs, "n_params": int(booster.num_trees()) * 15,
            "fit": {"best_round": booster.best_iteration, "seconds": time.time() - t0}, "extra": {}}
