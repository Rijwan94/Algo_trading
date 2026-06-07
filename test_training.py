import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.market_data_yfinance import fetch_yfinance_data
from src.data.fundamental_data import fetch_forexfactory_calendar, apply_news_impact_to_pair
from src.features.engineering import add_technical_features, merge_correlated_assets, create_multi_timeframe_features
from src.models.train import create_target_variable, prepare_data_for_training, train_xgboost_model, save_model

SYMBOLS_TO_TRAIN = ["XAUUSD", "XAGUSD", "XAUCHF", "EURUSD"]
TIMEFRAME = "15m"
HTF_TIMEFRAME = "1h"

# User requested features: DXY, US 10y treasury yield, S&P 500, VIX, US 30, USTEC
CORRELATED_ASSETS = ["DXY", "US10Y", "SP500", "VIX", "US30", "USTEC"]

# Fetch news once
print("Fetching fundamental news...")
start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
end = datetime.now().strftime('%Y-%m-%d')
df_news = fetch_forexfactory_calendar(start, end)

print("Fetching correlated asset data...")
corr_dfs = []
corr_suffixes = []
for asset in CORRELATED_ASSETS:
    df_asset = fetch_yfinance_data(asset, interval=TIMEFRAME, period="60d")
    if not df_asset.empty:
        corr_dfs.append(df_asset)
        corr_suffixes.append(asset)


for symbol in SYMBOLS_TO_TRAIN:
    print(f"\n=====================================")
    print(f"Training Pipeline for {symbol}")
    print(f"=====================================")

    print(f"Fetching primary data ({symbol} {TIMEFRAME})...")
    df = fetch_yfinance_data(symbol, TIMEFRAME, "60d") # YFinance max for 15m is 60d

    print(f"Fetching higher timeframe data ({symbol} {HTF_TIMEFRAME})...")
    df_1h = fetch_yfinance_data(symbol, HTF_TIMEFRAME, "60d")

    if df.empty or df_1h.empty:
        print(f"Skipping {symbol} due to missing data.")
        continue

    print("Adding technicals...")
    df = add_technical_features(df)

    print("Mapping higher timeframe context...")
    df = create_multi_timeframe_features(df, df_1h, prefix='HTF_')

    print("Merging Correlated Assets...")
    # NOTE: Some correlated assets only trade during specific market hours. Merging them and dropping NA
    # might result in dropping all rows if the asset market hours don't perfectly overlap
    # the forex 24/5 market hours. Instead of dropna inside merge, we ffill then bfill, then dropna.
    # The merge_correlated_assets function already ffills and then dropnas, which causes the zero shape issue.
    df = merge_correlated_assets(df, corr_dfs, corr_suffixes)

    if df.empty:
        print(f"Skipping {symbol} due to empty feature matrix after merging correlated assets.")
        continue

    print("Applying news impact...")
    df = apply_news_impact_to_pair(df, df_news, symbol)

    print("Creating target variable...")
    df = create_target_variable(df)

    if df.empty:
        print(f"Skipping {symbol} due to empty feature matrix after creating target variable.")
        continue

    print(f"Final Feature Matrix shape: {df.shape}")

    X, y = prepare_data_for_training(df)

    if X.empty:
        print(f"Skipping {symbol} due to empty feature matrix after processing.")
        continue

    model, acc = train_xgboost_model(X, y)
    save_model(model, symbol, TIMEFRAME)
