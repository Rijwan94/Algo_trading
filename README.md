# Algo_trading

ML/AI pipeline for forex and metals time-series forecasting with volatility- and volume-aware features, walk-forward evaluation, and risk-aware confidence scoring.

## Key Features
- Fetches historical data for forex/metals pairs (default: Yahoo Finance).
- Volatility, volume/tick proxy, session, and correlation features.
- Walk-forward training to reduce data leakage.
- Quantile regression to estimate prediction intervals and confidence.
- Backtesting with transaction costs and slippage.
- Model artifacts saved with metadata and registry.

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Confirm requirements**
   Update `configs/default.yaml` for:
   - prediction horizon and bar timeframe
   - supported pairs
   - data provider (default: yfinance)
   - accuracy/confidence metrics and thresholds

3. **Run training**
   ```bash
   PYTHONPATH=src python -m algo_trading.cli --config configs/default.yaml --acknowledge-risk
   ```

4. **Review outputs**
   - Model artifacts: `models/<pair>/<timestamp>/`
   - Registry: `models/model_registry.json`

## Notes on Data Providers
- `yfinance` supports many FX and metals symbols (e.g., `EURUSD=X`, `XAUUSD=X`).
- For providers without real volume, the pipeline uses a range-based proxy.
- To use CSV ingestion, set `data.provider: csv` and place files in `data/raw/` named `<PAIR>.csv`.

## Risk Disclosure
See [docs/RISK_DISCLOSURE.md](docs/RISK_DISCLOSURE.md). Real-money trading requires explicit acknowledgement.
