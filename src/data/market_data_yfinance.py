import yfinance as yf
import pandas as pd

# Mapping typical symbols to yfinance symbols
# Free Yahoo Finance is sometimes restrictive with specific metal/forex pairings,
# so we use Futures as a highly correlated proxy for the model training in the sandbox.
YF_SYMBOL_MAP = {
    # Forex
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "GBPJPY": "GBPJPY=X",
    "USDCHF": "CHF=X",

    # Metals (using Futures as proxy for YFinance)
    "XAUUSD": "GC=F", # Gold Futures
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

    # Special Synthetic pair handling for XAUCHF
    # XAUCHF = XAUUSD * USDCHF. Since YF doesn't reliable have XAUCHF=X, we synthesize it if needed.
    if symbol == "XAUCHF":
        df_xau = fetch_yfinance_data("XAUUSD", interval, period)
        df_chf = fetch_yfinance_data("USDCHF", interval, period)

        if df_xau.empty or df_chf.empty:
            return pd.DataFrame()

        # Align indexes
        df_xau, df_chf = df_xau.align(df_chf, join='inner')

        # Multiply to get synthetic OHLC
        df_synth = pd.DataFrame(index=df_xau.index)
        df_synth['open'] = df_xau['open'] * df_chf['open']
        df_synth['high'] = df_xau['high'] * df_chf['high']
        df_synth['low'] = df_xau['low'] * df_chf['low']
        df_synth['close'] = df_xau['close'] * df_chf['close']
        df_synth['tick_volume'] = df_xau['tick_volume'] # Proxy volume
        return df_synth

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
    df = fetch_yfinance_data("XAUUSD", interval="1h", period="7d")
    print("XAUUSD 1h data:")
    print(df.head())

    df2 = fetch_yfinance_data("XAUCHF", interval="1h", period="7d")
    print("XAUCHF 1h data:")
    print(df2.head())
