import yfinance as yf
import pandas as pd

# Mapping typical symbols to yfinance symbols
YF_SYMBOL_MAP = {
    # Forex
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X", # YFinance USDJPY=X is sometimes broken, JPY=X is USD/JPY
    "GBPJPY": "GBPJPY=X",
    "USDCHF": "CHF=X",

    # Metals
    "XAUUSD": "GC=F", # Gold Futures (often used as proxy on yfinance, or "XAUUSD=X" if available)
    "XAGUSD": "SI=F", # Silver Futures

    # Indices
    "DXY": "DX-Y.NYB", # US Dollar Index
    "US10Y": "^TNX", # 10-Year Treasury Yield
    "SP500": "^GSPC",
    "US30": "^DJI", # Dow Jones
    "USTEC": "^IXIC", # Nasdaq
    "VIX": "^VIX"
}

def fetch_yfinance_data(symbol, interval="1h", period="1y"):
    """
    Fetches historical OHLCV data from yfinance.

    Args:
        symbol (str): The common symbol name (e.g., "EURUSD", "XAUUSD").
        interval (str): Timeframe (e.g., "15m", "1h", "1d").
        period (str): Data period (e.g., "60d", "1y", "max").

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with datetime index.
    """
    yf_symbol = YF_SYMBOL_MAP.get(symbol, symbol)

    ticker = yf.Ticker(yf_symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        print(f"Warning: No data found for {symbol} ({yf_symbol})")
        return pd.DataFrame()

    # Clean up dataframe
    df.index.name = "time"
    df.index = pd.to_datetime(df.index)

    # Drop irrelevant columns if they exist
    cols_to_drop = ["Dividends", "Stock Splits", "Capital Gains"]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "tick_volume" # YFinance volume is proxy for tick volume in forex
    }, inplace=True)

    return df

if __name__ == "__main__":
    # Test fetch
    df = fetch_yfinance_data("EURUSD", interval="1h", period="7d")
    print("EURUSD 1h data:")
    print(df.head())
