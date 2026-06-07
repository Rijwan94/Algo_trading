import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import os

def create_target_variable(df, risk_reward=2.0):
    """
    Creates target variable: 1 (Up/Buy) or 0 (Down/Sell).
    For simplification, we look N periods ahead to see if the price moved
    up or down. In a more advanced setup with ATR, we simulate a trade
    checking if TP hits before SL.

    Here, we use a simple N-period forward return.
    """
    if df.empty:
        return df

    df = df.copy()

    # Predict direction N periods ahead (e.g., 3 periods)
    lookahead = 3
    df['future_return'] = df['close'].shift(-lookahead) - df['close']

    # Binary classification target
    # 1 if price goes up, 0 if price goes down
    df['target'] = np.where(df['future_return'] > 0, 1, 0)

    # Drop rows where target is NaN (the last 'lookahead' rows)
    df.dropna(subset=['target'], inplace=True)

    return df

def prepare_data_for_training(df, target_col='target'):
    """
    Separates features and targets, and drops raw price columns
    that shouldn't be used for predicting to avoid data leakage.
    """
    # Columns to strictly exclude from training features
    exclude_cols = ['open', 'high', 'low', 'close', 'future_return', target_col]

    # Select feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols]
    y = df[target_col]

    return X, y

def train_xgboost_model(X, y):
    """
    Trains an XGBoost Classifier on the provided data.
    Uses time-series splitting (no shuffling) to prevent future data leakage.
    """
    # Time-series split (train on past, test on future)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Training Complete.")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model, accuracy

def save_model(model, symbol, timeframe, path="models"):
    """Saves the trained model to disk."""
    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{path}/xgb_model_{symbol}_{timeframe}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def load_model(symbol, timeframe, path="models"):
    """Loads a trained model from disk."""
    filename = f"{path}/xgb_model_{symbol}_{timeframe}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    print(f"Model not found at {filename}")
    return None
