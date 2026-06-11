from src.data.market_data_yfinance import fetch_yfinance_data
from src.features.engineering import add_technical_features, merge_correlated_assets

df = fetch_yfinance_data("EURUSD", "1h", "60d")
df_tech = add_technical_features(df)
print(f"Technical features shape: {df_tech.shape}")
print(df_tech.columns)

df_dxy = fetch_yfinance_data("DXY", "1h", "60d")
df_merged = merge_correlated_assets(df_tech, [df_dxy], ["DXY"])
print(f"Merged features shape: {df_merged.shape}")
print(df_merged.columns)
