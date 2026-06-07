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

# Configuration
SYMBOL = "EURUSD"
TIMEFRAME = "15m"
HTF_TIMEFRAME = "1h"
RISK_REWARD = 2.0
ACCOUNT_RISK_PERCENT = 1.0
SIMULATED_BALANCE = 10000.0

def run_bot_cycle():
    print(f"--- Running Cycle for {SYMBOL} ---")

    # 1. Load Model
    model = load_model(SYMBOL, TIMEFRAME)
    if model is None:
        print("Model not found. Please train first.")
        return

    # 2. Fetch Latest Data
    df = fetch_yfinance_data(SYMBOL, interval=TIMEFRAME, period="10d")
    df_1h = fetch_yfinance_data(SYMBOL, interval=HTF_TIMEFRAME, period="10d")
    df_dxy = fetch_yfinance_data("DXY", interval=TIMEFRAME, period="10d")

    start = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    df_news = fetch_forexfactory_calendar(start, end)

    if df.empty:
        print("Failed to fetch primary data.")
        return

    # 3. Engineer Full Feature Matrix
    df_features = add_technical_features(df)
    df_features = create_multi_timeframe_features(df_features, df_1h, prefix='HTF_')
    df_features = merge_correlated_assets(df_features, [df_dxy], ["DXY"])
    df_features = apply_news_impact_to_pair(df_features, df_news, SYMBOL)

    if df_features.empty:
        print("Feature engineering resulted in empty dataframe (likely not enough data).")
        return

    # 4. Prepare latest row for prediction
    latest_row = df_features.iloc[-1:].copy()
    current_price = latest_row['close'].values[0]
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

    # 6. Risk Management
    sl, tp = calculate_dynamic_risk(current_price, current_atr, prediction, RISK_REWARD)
    volume = calculate_position_size(SIMULATED_BALANCE, ACCOUNT_RISK_PERCENT, current_price, sl)

    print(f"Analysis complete on full multi-timeframe & fundamental feature set.")
    print(f"Price: {current_price:.5f}, ATR: {current_atr:.5f}")
    print(f"Prediction: {'BUY/UP' if prediction == 1 else 'SELL/DOWN'}")
    print(f"Risk: {ACCOUNT_RISK_PERCENT}% -> Lot Size: {volume}")
    print(f"SL: {sl:.5f}, TP: {tp:.5f}")

    # 7. Execute
    place_order(SYMBOL, prediction, volume, sl, tp)
    print("Cycle complete.")

if __name__ == "__main__":
    run_bot_cycle()
