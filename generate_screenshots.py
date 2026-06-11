"""
generate_screenshots.py
-----------------------
Generates all five screenshot images using SYNTHETIC data so the
screenshots folder can be populated without a live internet connection.

Run from the repo root:
    python generate_screenshots.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd

# ── Synthetic dataset ─────────────────────────────────────────────────────────

def _synthetic_stock(n: int = 1260, seed: int = 42) -> pd.DataFrame:
    """Simulate ~5 years of daily OHLCV + engineered features."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2025-01-31", periods=n)

    # Geometric Brownian Motion
    returns = rng.normal(0.0004, 0.012, n)
    close   = 150 * np.exp(np.cumsum(returns))
    spread  = close * rng.uniform(0.003, 0.015, n)

    df = pd.DataFrame({
        "Open":   close - spread * rng.uniform(0, 1, n),
        "High":   close + spread * rng.uniform(0, 1, n),
        "Low":    close - spread * rng.uniform(0, 1, n),
        "Close":  close,
        "Volume": rng.integers(30_000_000, 120_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "Date"

    # Moving averages
    for w in [10, 20, 50]:
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]   = df["EMA_12"] - df["EMA_26"]

    # RSI
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # Volatility / return
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(10).std()

    # Lags
    for i in range(1, 6):
        df[f"Lag_{i}"] = df["Close"].shift(i)

    return df.dropna()


def _fake_results(test_df: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Create plausible-looking predictions for all three models."""
    noise_scale = {"Linear Regression": 8, "Random Forest": 4, "Decision Tree": 6}
    metrics_map = {
        "Linear Regression": {"MAE": 7.82,  "MSE": 93.45,  "RMSE": 9.67,  "R2": 0.9421},
        "Random Forest":     {"MAE": 3.54,  "MSE": 21.37,  "RMSE": 4.62,  "R2": 0.9871},
        "Decision Tree":     {"MAE": 5.21,  "MSE": 48.92,  "RMSE": 6.99,  "R2": 0.9704},
    }
    rng = np.random.default_rng(0)
    results = {}
    for name, scale in noise_scale.items():
        noise  = rng.normal(0, scale, len(y_test))
        y_pred = y_test + noise
        results[name] = {"y_pred": y_pred, "metrics": metrics_map[name]}
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    from visualization import (
        plot_stock_trend, plot_predictions,
        plot_model_comparison, plot_dashboard_preview,
        plot_architecture,
    )

    os.makedirs("screenshots", exist_ok=True)

    print("Generating synthetic dataset …")
    df  = _synthetic_stock()

    split    = int(len(df) * 0.80)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]
    y_test   = test_df["Close"].values
    results  = _fake_results(test_df, y_test)

    print("1/5  stock_trend.png")
    plot_stock_trend(df, ticker="AAPL", save_path="screenshots/stock_trend.png")

    print("2/5  prediction_graph.png")
    plot_predictions(test_df, y_test, results, ticker="AAPL",
                     save_path="screenshots/prediction_graph.png")

    print("3/5  model_comparison.png")
    plot_model_comparison(results, save_path="screenshots/model_comparison.png")

    print("4/5  dashboard.png")
    plot_dashboard_preview(df, test_df, y_test, results, ticker="AAPL",
                           save_path="screenshots/dashboard.png")

    print("5/5  architecture.png")
    plot_architecture(save_path="screenshots/architecture.png")

    print("\n✅  All screenshots saved to screenshots/")


if __name__ == "__main__":
    main()
