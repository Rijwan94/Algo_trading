from __future__ import annotations

from typing import Dict, Optional

import time

import pandas as pd
import yfinance as yf

from .utils import ensure_dir


def _standardize_dataframe(df: pd.DataFrame, timezone: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone)
    else:
        df.index = df.index.tz_convert(timezone)
    df = df.sort_index()
    return df


def fetch_yfinance_pair(
    pair: str,
    start: str,
    end: Optional[str],
    interval: str,
    timezone: str,
    retries: int = 3,
    backoff: float = 2.0,
) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    df = pd.DataFrame()
    for attempt in range(retries):
        try:
            df = yf.download(pair, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
            if not df.empty:
                break
        except Exception as exc:
            last_error = exc
        time.sleep(backoff ** attempt)

    if df.empty:
        message = f"No data returned for {pair}. Check symbol or provider availability."
        if last_error is not None:
            message = f"{message} Last error: {last_error}"
        raise ValueError(message)
    df = _standardize_dataframe(df, timezone)
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[required_cols]
    df = df.dropna(subset=["close"])
    return df


def load_csv_pair(path: str, timezone: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0])
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df = df.set_index("timestamp")
    df = _standardize_dataframe(df, timezone)
    return df


def fetch_all_pairs(config: Dict) -> Dict[str, pd.DataFrame]:
    provider = config["data"]["provider"]
    pairs = config["data"]["pairs"]
    start = config["data"]["start"]
    end = config["data"].get("end")
    interval = config["data"]["interval"]
    timezone = config["preprocessing"].get("timezone", "UTC")
    raw_dir = config["data"]["storage"]["raw_dir"]
    save_csv = config["data"]["ingestion"].get("save_csv", True)

    data: Dict[str, pd.DataFrame] = {}
    for pair in pairs:
        if provider == "yfinance":
            df = fetch_yfinance_pair(pair, start, end, interval, timezone)
        elif provider == "csv":
            df = load_csv_pair(f"{raw_dir}/{pair}.csv", timezone)
        else:
            raise ValueError(f"Unsupported data provider: {provider}")

        if save_csv:
            ensure_dir(raw_dir)
            df.to_csv(f"{raw_dir}/{pair}.csv")

        data[pair] = df
    return data


def align_pairs(pair_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if not pair_data:
        return pair_data

    common_index = None
    for df in pair_data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    aligned = {}
    for pair, df in pair_data.items():
        aligned[pair] = df.loc[common_index].copy()

    return aligned
