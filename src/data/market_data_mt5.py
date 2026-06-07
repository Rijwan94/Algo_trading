import pandas as pd
from datetime import datetime
import pytz

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 library not found. MT5 fetching will be disabled.")


def init_mt5():
    """Initializes connection to MT5 terminal."""
    if not MT5_AVAILABLE:
        print("MetaTrader5 not available in this environment.")
        return False

    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        mt5.shutdown()
        return False
    return True


def fetch_mt5_data(symbol, timeframe=mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None, num_candles=5000):
    """
    Fetches historical OHLCV data from MT5.

    Args:
        symbol (str): Symbol name (e.g., "EURUSD").
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1).
        num_candles (int): Number of candles to fetch.

    Returns:
        pd.DataFrame: DataFrame containing OHLCV data.
    """
    if not init_mt5():
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
    mt5.shutdown()

    if rates is None:
        print(f"Failed to fetch data for {symbol}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time in seconds into the datetime format
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Clean up
    df = df[['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]

    return df
