from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def build_training_frame(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    frame = df.copy()
    frame["target"] = frame["close"].pct_change(horizon).shift(-horizon)
    frame = frame.dropna()
    return frame


def select_features(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    numeric_frame = frame.select_dtypes(include=["number"]).copy()
    y = numeric_frame.pop("target")
    return numeric_frame, y


def fit_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: List[float],
    params: Dict,
) -> Dict[float, GradientBoostingRegressor]:
    models = {}
    for quantile in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            n_estimators=params.get("n_estimators", 300),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 3),
            random_state=42,
        )
        model.fit(X, y)
        models[quantile] = model
    return models


def predict_quantiles(models: Dict[float, GradientBoostingRegressor], X: np.ndarray) -> Dict[str, np.ndarray]:
    predictions = {}
    for quantile, model in models.items():
        key = f"q{int(quantile * 100)}"
        predictions[key] = model.predict(X)
    return predictions


def scale_features(
    X_train: pd.DataFrame,
    X_other: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_other_scaled = scaler.transform(X_other)
    return X_train_scaled, X_other_scaled, scaler
