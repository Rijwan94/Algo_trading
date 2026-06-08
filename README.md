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

## Running the Bot Automatically on your Laptop

To run this completely automated on your Windows laptop connected to your MetaTrader 5 account:

1. **Install Python & MetaTrader 5**
   - Make sure you have Python installed on your Windows laptop.
   - Download and log into the MetaTrader 5 application. Keep it open running in the background.

2. **Install Dependencies**
   - Open Command Prompt or PowerShell in this project folder.
   - Run: `pip install -r requirements.txt`
   - Run: `pip install MetaTrader5`

3. **Train the Models First**
   - The bot needs the models built on your machine first.
   - Run: `python test_training.py`
   - Wait for it to fetch data and save the `.pkl` files to the `models/` folder.

4. **Start the Fully Automated Bot**
   - Run: `python run_automated.py`
   - The script will automatically calculate the exact time until the next 15-minute candle closes (e.g., waiting for 10:15, 10:30, 10:45) and will automatically execute `src/main.py` at exactly the right time all day long.
   - Leave the command prompt window open.
