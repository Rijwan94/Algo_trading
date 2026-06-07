# End-to-End Trading ML Pipeline

This project builds a full pipeline to trade high-volatility pairs (Forex, Gold, Silver) using Machine Learning based on Technical Analysis, Multi-Timeframe Alignment, Correlated Assets (DXY, VIX, US10Y, SP500, US30, USTEC), and Fundamental News Impact.

## Setup

1. Install requirements:
`pip install -r requirements.txt`

2. Train the model:
`export PYTHONPATH=$PYTHONPATH:$(pwd) && python test_training.py`
This generates `.pkl` model files for the specified symbols (e.g. XAUUSD, XAGUSD, XAUCHF, EURUSD) by pulling in all multi-timeframe and news data. Models are excluded from git.

3. Run the bot cycle (Execution):
`export PYTHONPATH=$PYTHONPATH:$(pwd) && python src/main.py`
This runs predictions on the latest live data and simulates an MT5 trade using a dynamic 1:2 ATR-based Risk/Reward ratio.

*Note: The actual MetaTrader 5 module will automatically activate if run on a Windows machine with MT5 installed and python-metatrader5 package configured.*
