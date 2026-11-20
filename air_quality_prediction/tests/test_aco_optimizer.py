import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aco_optimizer import ACO_FeatureSelection, feature_category  # noqa: E402


def test_feature_category_handles_engineered_names():
    assert feature_category("PM2.5") == "policy"
    assert feature_category("PM2.5_lag1") == "policy"
    assert feature_category("NO2_avg7") == "policy"
    assert feature_category("NOx_lag3") == "policy"
    assert feature_category("temperature") == "weather"
    assert feature_category("wind_speed") == "weather"
    assert feature_category("AQI_lag1") == "other"
    assert feature_category("month_sin") == "other"


def _data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 12)),
                     columns=["PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
                              "temperature", "humidity", "noise1", "noise2", "noise3", "noise4"])
    y = 3 * X["PM2.5"] + 2 * X["NO2"] + X["temperature"] + rng.normal(scale=0.1, size=n)
    return X.iloc[:400], y.iloc[:400], X.iloc[400:], y.iloc[400:]


def test_optimize_finds_informative_features_and_is_reproducible():
    Xtr, ytr, Xva, yva = _data()
    kw = dict(n_ants=10, n_iterations=15, min_features=2, max_features=5, random_state=1)
    r1 = ACO_FeatureSelection(**kw).optimize(Xtr, ytr, Xva, yva)
    r2 = ACO_FeatureSelection(**kw).optimize(Xtr, ytr, Xva, yva)
    assert r1["best_features"] == r2["best_features"]
    assert {"PM2.5", "NO2"} <= set(r1["best_features"])
    assert 2 <= len(r1["best_features"]) <= 5


def test_max_features_reachable_and_rerun_resets_state():
    Xtr, ytr, Xva, yva = _data()
    aco = ACO_FeatureSelection(n_ants=40, n_iterations=3, min_features=3, max_features=3,
                               patience=100)
    aco.optimize(Xtr, ytr, Xva, yva)
    assert all(len(h["selected_features"]) == 3 for h in aco.feature_selection_history)
    aco.optimize(Xtr, ytr, Xva, yva)
    assert len(aco.convergence_history) == 3
