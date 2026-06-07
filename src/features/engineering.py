import pandas as pd
import numpy as np
import ta

def add_technical_features(df):
    """
    Adds technical indicators to the OHLCV dataframe using the 'ta' library.
    Includes momentum, volatility, volume, and trend indicators.
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. Volatility Features
    # Average True Range (ATR) - Crucial for dynamic TP/SL
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['bb_high'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['bb_width'] = ta.volatility.bollinger_wband(df['close'], window=20, window_dev=2)

    # 2. Trend Features
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200)
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_diff'] = ta.trend.macd_diff(df['close'])
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    # 3. Momentum Features
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=14, smooth_window=3)

    # 4. Custom Price Action Features
    # Candlestick body and wicks
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['candle_range'] = df['high'] - df['low']

    # Price returns (momentum)
    df['return_1p'] = df['close'].pct_change(1)
    df['return_3p'] = df['close'].pct_change(3)

    # Drop rows with NaN values created by window functions
    df.dropna(inplace=True)

    return df

def merge_correlated_assets(df_main, related_dfs, suffixes):
    """
    Merges related asset features (like DXY, VIX, US10Y) into the main dataframe.
    """
    if df_main.empty:
        return df_main

    df_merged = df_main.copy()

    for rel_df, suffix in zip(related_dfs, suffixes):
        if rel_df.empty:
            continue

        # We only want to join the close price and maybe its return to avoid feature bloat
        rel_feat = pd.DataFrame(index=rel_df.index)
        rel_feat[f'close_{suffix}'] = rel_df['close']
        rel_feat[f'return_{suffix}'] = rel_df['close'].pct_change(1)

        # Convert index timezone of related dataframe to match main dataframe
        if rel_feat.index.tz is not None and df_main.index.tz is not None:
            rel_feat.index = rel_feat.index.tz_convert(df_main.index.tz)
        elif rel_feat.index.tz is None and df_main.index.tz is not None:
            rel_feat.index = rel_feat.index.tz_localize(df_main.index.tz)

        # Merge using merge_asof because timestamps from indices and futures
        # may not align exactly to the minute/hour of forex pairs
        df_merged = df_merged.sort_index()
        rel_feat = rel_feat.sort_index()

        # Ensure index isn't duplicated
        rel_feat = rel_feat[~rel_feat.index.duplicated(keep='last')]
        df_merged = df_merged[~df_merged.index.duplicated(keep='last')]

        df_merged = pd.merge_asof(
            df_merged,
            rel_feat,
            left_index=True,
            right_index=True,
            direction='backward', # Use latest available data without lookahead
            tolerance=pd.Timedelta('12h') # Allow up to 12 hours staleness (e.g. overnight)
        )

        # For correlated assets that start trading later in the day or week,
        # ffill will fail for the first few rows, leaving NaNs which destroy the whole dataframe on dropna.
        # We must ffill and then bfill, or fill with 0 to prevent total data loss.
        df_merged[f'close_{suffix}'] = df_merged[f'close_{suffix}'].ffill().bfill()
        df_merged[f'return_{suffix}'] = df_merged[f'return_{suffix}'].ffill().bfill().fillna(0)

    df_merged.dropna(inplace=True)
    return df_merged

def create_multi_timeframe_features(df_lower, df_higher, prefix='HTF_'):
    """
    Merges higher timeframe features into a lower timeframe execution chart.
    E.g., mapping 1H EMA trend onto a 15M chart.
    """
    # For simplification, we calculate standard features on HTF, then merge them.
    df_htf_feats = add_technical_features(df_higher)

    # Select key HTF features
    cols_to_keep = ['ema_21', 'ema_50', 'rsi_14', 'macd_diff', 'atr_14']
    df_htf_reduced = df_htf_feats[cols_to_keep].add_prefix(prefix)

    # SHIFT HTF features by 1 period to avoid lookahead bias!
    # A candle labeled 10:00 closes at 10:59, so its data cannot be known at 10:15 or 10:30.
    # By shifting the HTF index forward by 1 period (the size of the HTF candle), we ensure
    # we only join the *most recently closed* HTF data to our current LTF timestamp.
    df_htf_shifted = df_htf_reduced.shift(1)

    # Merge, forward filling
    df_merged = df_lower.join(df_htf_shifted, how='left')
    df_merged.ffill(inplace=True)
    df_merged.dropna(inplace=True)

    return df_merged
