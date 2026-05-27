# Usage

## Configuration
Edit `configs/default.yaml` to align with your requirements:

- **Prediction horizon**: `model.horizon` (number of bars ahead)
- **Timeframe**: `data.interval` (e.g., 1h, 4h, 1d)
- **Pairs**: `data.pairs`
- **Provider**: `data.provider` (`yfinance` or `csv`)
- **Confidence**: `confidence.minimum_confidence` and `confidence.target_coverage`

## Pipeline Steps
1. **Ingestion**: fetches and stores raw data in `data/raw/`.
2. **Preprocessing**: missing data, outlier clipping, timezone alignment.
3. **Features**: returns, volatility, volume proxy, market session flags, pair correlations.
4. **Training**: quantile gradient boosting models for prediction intervals.
5. **Evaluation**: walk-forward metrics and interval coverage checks.
6. **Backtest**: confidence-filtered strategy with cost assumptions.
7. **Save**: models, scaler, and metrics stored in `models/` with registry entries.

## Running
```bash
PYTHONPATH=src python -m algo_trading.cli --config configs/default.yaml --acknowledge-risk
```

## Outputs
- `models/<pair>/<timestamp>/model_q10.joblib` (lower quantile)
- `models/<pair>/<timestamp>/model_q50.joblib` (median)
- `models/<pair>/<timestamp>/model_q90.joblib` (upper quantile)
- `models/<pair>/<timestamp>/scaler.joblib`
- `models/<pair>/<timestamp>/metrics.json`

## Risk Acknowledgement
The CLI requires `--acknowledge-risk` when `risk.require_confirmation` is true.
