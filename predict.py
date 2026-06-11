"""
predict.py
----------
Runs the full prediction pipeline end-to-end and returns
predictions + metrics for all three models.
"""

import sys
import os

# Allow running from repo root OR from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data_collection import fetch_stock_data, save_stock_data
from preprocessing   import preprocess
from train_model     import train_all_models


def run_prediction(ticker: str = "AAPL", period: str = "5y"):
    """
    Full pipeline: fetch → preprocess → train → predict.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol.
    period : str
        Historical period (e.g. '2y', '5y').

    Returns
    -------
    results  : dict    (see train_model.train_all_models)
    test_df  : pd.DataFrame   (test-split rows, index = Date)
    y_test   : np.ndarray
    """
    # 1. Collect
    df = fetch_stock_data(ticker, period)
    save_stock_data(df, "data/stock_data.csv")

    # 2. Pre-process + feature engineering
    X_train, X_test, y_train, y_test, train_df, test_df = preprocess(df)

    # 3. Train all models
    results = train_all_models(X_train, X_test, y_train, y_test)

    # 4. Summary table
    print("\n" + "=" * 65)
    print(f"  MODEL COMPARISON  —  {ticker}  ({period})")
    print("=" * 65)
    header = f"{'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print(header)
    print("-" * 65)
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<25} {m['MAE']:>8.3f} {m['RMSE']:>8.3f} {m['R2']:>8.4f}")
    print("=" * 65)

    return results, test_df, y_test


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_prediction(ticker, "5y")
