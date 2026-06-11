import time
import os
import sys
from datetime import datetime, timedelta

# Ensure Python path includes the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_data_yfinance import fetch_yfinance_data
from src.data.fundamental_data import fetch_forexfactory_calendar, apply_news_impact_to_pair
from src.features.engineering import add_technical_features, merge_correlated_assets, create_multi_timeframe_features
from src.models.train import load_model, prepare_data_for_training
from src.execution.risk_manager import calculate_dynamic_risk, calculate_position_size
from src.execution.mt5_trader import place_order

# Try to import MT5 for actual pricing when live
try:
    import MetaTrader5 as mt5
    from src.data.market_data_mt5 import fetch_mt5_data, init_mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Configuration
SYMBOLS_TO_TRADE = ["XAUUSD", "XAGUSD", "XAUCHF", "EURUSD"]
TIMEFRAME = "15m"
HTF_TIMEFRAME = "1h"
RISK_REWARD = 2.0
ACCOUNT_RISK_PERCENT = 1.0
SIMULATED_BALANCE = 10000.0

def run_bot_cycle():
    print(f"--- Running Bot Cycle at {datetime.now()} ---")

    # User requested features: DXY, US 10y treasury yield, S&P 500, VIX, US 30, USTEC
    CORRELATED_ASSETS = ["DXY", "US10Y", "SP500", "VIX", "US30", "USTEC"]

    # Pre-fetch global data to save time in loop
    corr_dfs = []
    corr_suffixes = []
    for asset in CORRELATED_ASSETS:
        df_asset = fetch_yfinance_data(asset, interval=TIMEFRAME, period="60d")
        if not df_asset.empty:
            corr_dfs.append(df_asset)
            corr_suffixes.append(asset)

    start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    df_news = fetch_forexfactory_calendar(start, end)

    for symbol in SYMBOLS_TO_TRADE:
        print(f"\nProcessing {symbol}...")

        # 1. Load Model
        model = load_model(symbol, TIMEFRAME)
        if model is None:
            continue

        # 2. Fetch Latest Data
        df = fetch_yfinance_data(symbol, interval=TIMEFRAME, period="60d")
        df_1h = fetch_yfinance_data(symbol, interval=HTF_TIMEFRAME, period="60d")

        if df.empty or df_1h.empty:
            print(f"Failed to fetch data for {symbol}.")
            continue

        # 3. Engineer Full Feature Matrix
        df_features = add_technical_features(df)
        df_features = create_multi_timeframe_features(df_features, df_1h, prefix='HTF_')
        df_features = merge_correlated_assets(df_features, corr_dfs, corr_suffixes)
        df_features = apply_news_impact_to_pair(df_features, df_news, symbol)

        if df_features.empty:
            print(f"Feature engineering resulted in empty dataframe for {symbol}.")
            continue

        # 4. Prepare latest row for prediction
        latest_row = df_features.iloc[-1:].copy()
        current_atr = latest_row['atr_14'].values[0]

        # Add dummy target col for the preparation function so it doesn't fail
        latest_row['target'] = 0

        # Exclude price columns for prediction
        X, _ = prepare_data_for_training(latest_row)

        # Ensure columns match model
        expected_cols = model.get_booster().feature_names
        for col in expected_cols:
            if col not in X.columns:
                X[col] = 0.0 # Fallback
        X = X[expected_cols]

        # 5. Predict
        prediction = int(model.predict(X)[0]) # 1 for Buy, 0 for Sell

        # 6. Get REAL Execution Price (Crucial Fix for MT5 live vs YFinance Proxy Futures)
        if MT5_AVAILABLE and init_mt5():
            # Get real spot price from terminal, not futures data
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                current_price = tick.ask if prediction == 1 else tick.bid
                mt5.shutdown()
            else:
                print(f"Failed to get tick from MT5 for {symbol}, falling back to proxy price.")
                current_price = latest_row['close'].values[0]
                mt5.shutdown()
        else:
            current_price = latest_row['close'].values[0]

        # 7. Risk Management
        sl, tp = calculate_dynamic_risk(current_price, current_atr, prediction, RISK_REWARD)
        volume = calculate_position_size(SIMULATED_BALANCE, ACCOUNT_RISK_PERCENT, current_price, sl)

        print(f"[{symbol}] Price: {current_price:.5f}, ATR: {current_atr:.5f}")
        print(f"[{symbol}] Prediction: {'BUY/UP' if prediction == 1 else 'SELL/DOWN'}")
        print(f"[{symbol}] Risk: {ACCOUNT_RISK_PERCENT}% -> Lot Size: {volume}")
        print(f"[{symbol}] SL: {sl:.5f}, TP: {tp:.5f}")

        # 8. Execute
        place_order(symbol, prediction, volume, sl, tp)

    print("\n--- Cycle Complete ---")

if __name__ == "__main__":
    run_bot_cycle()
