from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .modeling import fit_quantile_models, predict_quantiles, scale_features


def interval_to_periods_per_year(interval: str) -> int:
    interval = interval.lower().strip()
    if interval.endswith("h"):
        hours = int(interval.replace("h", ""))
        return max(int((24 / hours) * 252), 1)
    if interval.endswith("d"):
        days = int(interval.replace("d", ""))
        return max(int(252 / days), 1)
    if interval.endswith("wk"):
        weeks = int(interval.replace("wk", ""))
        return max(int(52 / weeks), 1)
    if interval.endswith("mo"):
        months = int(interval.replace("mo", ""))
        return max(int(12 / months), 1)
    return 252


def walk_forward_splits(
    n_samples: int,
    initial_train_size: float,
    validation_size: float,
    test_size: float,
    step_size: float,
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    train_end = int(n_samples * initial_train_size)
    val_size = int(n_samples * validation_size)
    test_size_abs = int(n_samples * test_size)
    step = max(int(n_samples * step_size), 1)

    while train_end + val_size + test_size_abs <= n_samples:
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, train_end + val_size)
        test_idx = np.arange(train_end + val_size, train_end + val_size + test_size_abs)
        yield train_idx, val_idx, test_idx
        train_end += step


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(((y_true >= lower) & (y_true <= upper)).mean())


def coverage_error(coverage: float, target: float) -> float:
    return float(abs(coverage - target))


def compute_confidence(
    lower: np.ndarray,
    upper: np.ndarray,
    width_percentile: float,
) -> np.ndarray:
    width = upper - lower
    threshold = np.nanquantile(width, width_percentile)
    threshold = max(threshold, 1e-9)
    confidence = 1.0 - (width / threshold)
    return np.clip(confidence, 0.0, 1.0)


def evaluate_splits(
    X: pd.DataFrame,
    y: pd.Series,
    quantiles: List[float],
    params: Dict,
    split_cfg: Dict,
    confidence_cfg: Dict,
) -> Dict:
    metrics = []
    coverage_target = confidence_cfg.get("target_coverage", 0.8)
    width_percentile = confidence_cfg.get("width_percentile", 0.9)

    for train_idx, _, test_idx in walk_forward_splits(
        len(X),
        split_cfg["initial_train_size"],
        split_cfg["validation_size"],
        split_cfg["test_size"],
        split_cfg["step_size"],
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
        models = fit_quantile_models(X_train_scaled, y_train.values, quantiles, params)
        preds = predict_quantiles(models, X_test_scaled)

        lower = preds.get("q10")
        median = preds.get("q50")
        upper = preds.get("q90")

        coverage = interval_coverage(y_test.values, lower, upper)
        metrics.append(
            {
                "mae": mean_absolute_error(y_test, median),
                "rmse": mean_squared_error(y_test, median, squared=False),
                "r2": r2_score(y_test, median),
                "directional_accuracy": directional_accuracy(y_test.values, median),
                "interval_coverage": coverage,
                "coverage_error": coverage_error(coverage, coverage_target),
            }
        )

    if not metrics:
        return {}

    return {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}


def backtest_strategy(
    y_true: np.ndarray,
    median_pred: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    periods_per_year: int,
) -> Dict:
    position = 0.0
    returns = []
    cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0

    for actual, pred, conf in zip(y_true, median_pred, confidence):
        desired_position = 0.0
        if conf >= threshold:
            desired_position = 1.0 if pred > 0 else -1.0
        trade_cost = abs(desired_position - position) * cost_rate
        pnl = desired_position * actual - trade_cost
        returns.append(pnl)
        position = desired_position

    returns = np.array(returns)
    equity_curve = 1 + np.cumsum(returns)
    cumulative = float(equity_curve[-1] - 1) if len(equity_curve) else 0.0
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(periods_per_year))
    peak = np.maximum.accumulate(equity_curve) if len(equity_curve) else np.array([0.0])
    drawdown = peak - equity_curve if len(equity_curve) else np.array([0.0])
    max_drawdown = float(np.max(drawdown)) if len(drawdown) else 0.0

    return {
        "cumulative_return": cumulative,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
