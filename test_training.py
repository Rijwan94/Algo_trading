import sys
import os
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.market_data_yfinance import fetch_yfinance_data
from src.data.fundamental_data import fetch_forexfactory_calendar, apply_news_impact_to_pair
from src.features.engineering import add_technical_features, merge_correlated_assets, create_multi_timeframe_features
from src.models.train import create_target_variable, prepare_data_for_training, train_xgboost_model, save_model

print("Fetching primary data (EURUSD 15m)...")
df = fetch_yfinance_data("EURUSD", "15m", "60d") # YFinance max for 15m is 60d

print("Fetching higher timeframe data (EURUSD 1h)...")
df_1h = fetch_yfinance_data("EURUSD", "1h", "60d")

print("Fetching correlated asset data (DXY 15m)...")
df_dxy = fetch_yfinance_data("DXY", "15m", "60d")

print("Fetching fundamental news...")
start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
end = datetime.now().strftime('%Y-%m-%d')
df_news = fetch_forexfactory_calendar(start, end)

print("\nStarting Feature Engineering Pipeline...")

# 1. Base technicals on primary TF
print("Adding technicals...")
df = add_technical_features(df)

# 2. Multi-Timeframe mapping
print("Mapping higher timeframe context...")
df = create_multi_timeframe_features(df, df_1h, prefix='HTF_')

# 3. Correlated Assets
print("Merging DXY...")
df = merge_correlated_assets(df, [df_dxy], ["DXY"])

# 4. Fundamental Impact
print("Applying news impact...")
df = apply_news_impact_to_pair(df, df_news, "EURUSD")

# 5. Target Variable
df = create_target_variable(df)

print(f"\nFinal Feature Matrix shape: {df.shape}")
print("Features:", [c for c in df.columns if c not in ['target', 'open', 'high', 'low', 'close', 'future_return']])

X, y = prepare_data_for_training(df)
print(f"Features shape: {X.shape}, Target shape: {y.shape}")

model, acc = train_xgboost_model(X, y)
save_model(model, "EURUSD", "15m")
