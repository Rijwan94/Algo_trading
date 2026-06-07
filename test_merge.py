from src.data.market_data_yfinance import fetch_yfinance_data
from src.features.engineering import add_technical_features, create_multi_timeframe_features

df = fetch_yfinance_data("EURUSD", "15m", "60d")
df_1h = fetch_yfinance_data("EURUSD", "1h", "60d")

print("Original shape:", df.shape)

df = add_technical_features(df)
print("After technicals:", df.shape)

df = create_multi_timeframe_features(df, df_1h, prefix='HTF_')
print("After MTF:", df.shape)

from src.features.engineering import merge_correlated_assets
df_dxy = fetch_yfinance_data("DXY", "15m", "60d")
print("DXY shape:", df_dxy.shape)
df = merge_correlated_assets(df, [df_dxy], ["DXY"])
print("After DXY merge:", df.shape)

df_vix = fetch_yfinance_data("VIX", "15m", "60d")
print("VIX shape:", df_vix.shape)
df = merge_correlated_assets(df, [df_vix], ["VIX"])
print("After VIX merge:", df.shape)

df_us10y = fetch_yfinance_data("US10Y", "15m", "60d")
print("US10Y shape:", df_us10y.shape)
df = merge_correlated_assets(df, [df_us10y], ["US10Y"])
print("After US10Y merge:", df.shape)

df_sp500 = fetch_yfinance_data("SP500", "15m", "60d")
print("SP500 shape:", df_sp500.shape)
df = merge_correlated_assets(df, [df_sp500], ["SP500"])
print("After SP500 merge:", df.shape)
