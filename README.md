# End-to-End Trading ML Pipeline

This project builds a full pipeline to trade high-volatility pairs (Forex, Gold) using Machine Learning based on Technical Analysis, Multi-Timeframe Alignment, Correlated Assets (DXY, VIX), and Fundamental News Impact.

## Setup

1. Install requirements:
`pip install -r requirements.txt`

2. Train the model:
`python test_training.py`
This generates the `models/xgb_model_EURUSD_15m.pkl` file, pulling in all multi-timeframe and news data.

3. Run the bot cycle (Execution):
`python src/main.py`
This runs predictions on the latest live data and simulates an MT5 trade using a dynamic 1:2 ATR-based Risk/Reward ratio.

*Note: The actual MetaTrader 5 module will automatically activate if run on a Windows machine with MT5 installed and python-metatrader5 package configured.*
