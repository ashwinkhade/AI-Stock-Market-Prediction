"""
data_collection.py
------------------
Fetches historical stock data from Yahoo Finance using yfinance
and saves it to data/stock_data.csv.
"""

import os
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str = "AAPL", period: str = "5y") -> pd.DataFrame:
    """
    Download historical OHLCV data for a given ticker symbol.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL', 'TSLA', 'GOOGL').
    period : str
        Time period to download. Valid values: 1d, 5d, 1mo, 3mo,
        6mo, 1y, 2y, 5y, 10y, ytd, max.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, Dividends,
        Stock Splits. Index is a DatetimeIndex.
    """
    print(f"[INFO] Fetching {period} of data for ticker: {ticker}")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. "
            "Please check the symbol and try again."
        )

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    print(f"[INFO] Downloaded {len(df)} rows  ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def save_stock_data(df: pd.DataFrame, filepath: str = "data/stock_data.csv") -> None:
    """Save a stock DataFrame to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath)
    print(f"[INFO] Data saved to {filepath}")


def load_stock_data(filepath: str = "data/stock_data.csv") -> pd.DataFrame:
    """Load previously saved stock data from CSV."""
    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    print(f"[INFO] Loaded {len(df)} rows from {filepath}")
    return df


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "5y")
    save_stock_data(df)
    print(df.tail())
