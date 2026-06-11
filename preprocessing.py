"""
preprocessing.py
----------------
Cleans raw OHLCV data and engineers features for ML models.

Features created
----------------
MA_10, MA_20, MA_50   – simple moving averages
EMA_12, EMA_26        – exponential moving averages
MACD                  – MACD line (EMA12 – EMA26)
RSI                   – 14-period Relative Strength Index
Volatility            – 10-day rolling std of daily returns
Daily_Return          – percentage change in Close
Lag_1 … Lag_5         – lagged Close prices
"""

import numpy as np
import pandas as pd


# ── Feature engineering ───────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for window in [10, 20, 50]:
        df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = df["EMA_12"] - df["EMA_26"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_volatility(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(window).std()
    return df


def add_lag_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, lags + 1):
        df[f"Lag_{i}"] = df["Close"].shift(i)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_volatility(df)
    df = add_lag_features(df)
    return df


# ── Cleaning / splitting ──────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows introduced by rolling windows and lag shifts."""
    before = len(df)
    df = df.dropna()
    print(f"[INFO] Dropped {before - len(df)} rows with NaN  →  {len(df)} rows remain")
    return df


FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "MA_10", "MA_20", "MA_50",
    "EMA_12", "EMA_26", "MACD",
    "RSI", "Volatility", "Daily_Return",
    "Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5",
]
TARGET_COL = "Close"


def get_features_and_target(df: pd.DataFrame):
    """Return (X, y) arrays ready for sklearn."""
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    return X, y


def train_test_split_time(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Chronological train / test split (no shuffling).

    Returns
    -------
    train_df, test_df : pd.DataFrame
    """
    split = int(len(df) * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]


# ── Pipeline helper ───────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_df, test_df
    """
    df = engineer_features(df)
    df = clean_data(df)
    train_df, test_df = train_test_split_time(df, test_ratio)

    X_train, y_train = get_features_and_target(train_df)
    X_test,  y_test  = get_features_and_target(test_df)

    print(f"[INFO] Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    return X_train, X_test, y_train, y_test, train_df, test_df


if __name__ == "__main__":
    from data_collection import fetch_stock_data
    df = fetch_stock_data("AAPL", "5y")
    X_train, X_test, y_train, y_test, train_df, test_df = preprocess(df)
    print("Feature matrix shape:", X_train.shape)
