from src.data.market_data_yfinance import fetch_yfinance_data
from src.features.engineering import merge_correlated_assets

df = fetch_yfinance_data("EURUSD", "1h", "10d")
df_us10y = fetch_yfinance_data("US10Y", "1h", "10d")

print("EURUSD head:")
print(df.head(2))

print("US10Y head:")
print(df_us10y.head(2))
