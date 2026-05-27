from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import joblib

from .data import align_pairs, fetch_all_pairs
from .evaluation import (
    backtest_strategy,
    compute_confidence,
    evaluate_splits,
    interval_to_periods_per_year,
    walk_forward_splits,
)
from .features import add_correlation_features, build_features
from .modeling import build_training_frame, fit_quantile_models, predict_quantiles, scale_features, select_features
from .preprocessing import preprocess_pair
from .utils import ensure_dir, read_json, save_json


def run_pipeline(config: Dict) -> Dict:
    raw_data = fetch_all_pairs(config)
    aligned = align_pairs(raw_data)

    processed = {}
    for pair, df in aligned.items():
        processed[pair] = preprocess_pair(df, config)

    feature_frames = {pair: build_features(df, config) for pair, df in processed.items()}

    if len(feature_frames) > 1:
        feature_frames = add_correlation_features(
            feature_frames, config["features"]["correlations"]["window"]
        )

    results = {}
    for pair, frame in feature_frames.items():
        training_frame = build_training_frame(frame, config["model"]["horizon"])
        X, y = select_features(training_frame)

        metrics = evaluate_splits(
            X,
            y,
            config["model"]["quantiles"],
            config["model"]["hyperparameters"],
            config["preprocessing"]["train_validation_test"]["walk_forward"],
            config["confidence"],
        )

        results[pair] = {
            "rows": int(len(training_frame)),
            "metrics": metrics,
        }

        backtest = backtest_last_split(X, y, config)
        results[pair]["backtest"] = backtest

        saved = train_and_save_pair(pair, X, y, config, metrics)
        results[pair]["artifacts"] = saved

    return results


def train_and_save_pair(pair: str, X: pd.DataFrame, y: pd.Series, config: Dict, metrics: Dict) -> Dict:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(f"{config['models']['output_dir']}/{pair}/{timestamp}")

    X_scaled, _, scaler = scale_features(X)
    models = fit_quantile_models(
        X_scaled,
        y.values,
        config["model"]["quantiles"],
        config["model"]["hyperparameters"],
    )

    for quantile, model in models.items():
        joblib.dump(model, output_dir / f"model_q{int(quantile * 100)}.joblib")

    joblib.dump(scaler, output_dir / "scaler.joblib")

    metadata = {
        "pair": pair,
        "trained_at": timestamp,
        "rows": int(len(X)),
        "features": list(X.columns),
        "metrics": metrics,
        "config": {
            "model": config["model"],
            "preprocessing": config["preprocessing"],
            "features": config["features"],
        },
    }

    save_json(metadata, output_dir / "metrics.json")

    registry_path = config["models"]["registry_file"]
    registry = read_json(registry_path, default=[])
    registry.append(
        {
            "pair": pair,
            "trained_at": timestamp,
            "path": str(output_dir),
            "metrics": metrics,
        }
    )
    save_json(registry, registry_path)

    return {
        "model_dir": str(output_dir),
        "metrics_file": str(output_dir / "metrics.json"),
    }

def backtest_last_split(X: pd.DataFrame, y: pd.Series, config: Dict) -> Dict:
    periods_per_year = interval_to_periods_per_year(config["data"]["interval"])
    splits = list(
        walk_forward_splits(
            len(X),
            config["preprocessing"]["train_validation_test"]["walk_forward"]["initial_train_size"],
            config["preprocessing"]["train_validation_test"]["walk_forward"]["validation_size"],
            config["preprocessing"]["train_validation_test"]["walk_forward"]["test_size"],
            config["preprocessing"]["train_validation_test"]["walk_forward"]["step_size"],
        )
    )
    if not splits:
        return {}

    train_idx, _, test_idx = splits[-1]
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    models = fit_quantile_models(
        X_train_scaled,
        y_train.values,
        config["model"]["quantiles"],
        config["model"]["hyperparameters"],
    )
    preds = predict_quantiles(models, X_test_scaled)
    lower = preds.get("q10")
    median = preds.get("q50")
    upper = preds.get("q90")

    confidence = compute_confidence(lower, upper, config["confidence"]["width_percentile"])
    return backtest_strategy(
        y_true=y_test.values,
        median_pred=median,
        confidence=confidence,
        threshold=config["confidence"]["minimum_confidence"],
        transaction_cost_bps=config["backtest"]["transaction_cost_bps"],
        slippage_bps=config["backtest"]["slippage_bps"],
        periods_per_year=periods_per_year,
    )
