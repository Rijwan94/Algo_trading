from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def add_returns(df: pd.DataFrame, horizons: List[int]) -> pd.DataFrame:
    df = df.copy()
    for horizon in horizons:
        df[f"return_{horizon}"] = df["close"].pct_change(horizon)
    return df


def add_volatility(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        df[f"volatility_{window}"] = df["return_1"].rolling(window).std()
    return df


def add_volume_features(df: pd.DataFrame, use_volume: bool, proxy: str) -> pd.DataFrame:
    df = df.copy()
    if use_volume:
        volume = df["volume"].fillna(0.0)
        if proxy == "range":
            proxy_volume = (df["high"] - df["low"]).abs()
        else:
            proxy_volume = df["close"].diff().abs()
        volume = volume.mask(volume <= 0.0, proxy_volume)
        df["volume_proxy"] = proxy_volume
        df["volume_filled"] = volume
        df["volume_zscore"] = (volume - volume.rolling(50).mean()) / (volume.rolling(50).std() + 1e-9)
    return df


def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hours = df.index.hour
    df["session_asia"] = ((hours >= 0) & (hours < 8)).astype(int)
    df["session_europe"] = ((hours >= 7) & (hours < 15)).astype(int)
    df["session_us"] = ((hours >= 13) & (hours < 21)).astype(int)
    return df


def build_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    df = add_returns(df, config["features"]["returns"]["horizons"])
    df = add_volatility(df, config["features"]["volatility"]["windows"])
    df = add_volume_features(
        df,
        config["features"]["volume"]["use_volume"],
        config["features"]["volume"]["proxy"],
    )
    if config["features"]["sessions"]["enabled"]:
        df = add_session_flags(df)
    df["range"] = (df["high"] - df["low"]).abs()
    df["hlc3"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def add_correlation_features(pair_frames: Dict[str, pd.DataFrame], window: int) -> Dict[str, pd.DataFrame]:
    returns = {}
    for pair, frame in pair_frames.items():
        returns[pair] = frame["return_1"].rename(pair)

    returns_df = pd.concat(returns.values(), axis=1).dropna(how="all")

    updated = {}
    for pair, frame in pair_frames.items():
        frame = frame.copy()
        for other in returns_df.columns:
            if other == pair:
                continue
            correlation = returns_df[pair].rolling(window).corr(returns_df[other])
            frame[f"corr_{other}"] = correlation.reindex(frame.index)
        updated[pair] = frame

    return updated
