from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def handle_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ffill":
        return df.ffill().bfill()
    if method == "bfill":
        return df.bfill().ffill()
    if method == "drop":
        return df.dropna()
    return df


def clip_outliers(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    returns = df["close"].pct_change().replace([np.inf, -np.inf], np.nan)
    zscores = (returns - returns.mean()) / (returns.std(ddof=0) + 1e-9)
    mask = zscores.abs() > threshold
    df = df.copy()
    df.loc[mask, "close"] = np.nan
    return df


def preprocess_pair(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    df = handle_missing(df, config["preprocessing"]["missing"])

    outlier_cfg = config["preprocessing"].get("outlier", {})
    if outlier_cfg.get("method") == "zscore":
        df = clip_outliers(df, outlier_cfg.get("zscore_threshold", 5.0))
        df = handle_missing(df, config["preprocessing"]["missing"])

    return df
