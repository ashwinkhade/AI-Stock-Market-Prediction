"""
train_model.py
--------------
Trains Linear Regression, Random Forest Regressor, and Decision Tree
Regressor on the preprocessed stock data and reports evaluation metrics.
"""

import numpy as np
from sklearn.linear_model  import LinearRegression
from sklearn.ensemble       import RandomForestRegressor
from sklearn.tree           import DecisionTreeRegressor
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ── Metrics helper ────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  MSE  : {metrics['MSE']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  R²   : {metrics['R2']:.4f}")


# ── Individual trainers ───────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train, X_test, y_test):
    scaler  = StandardScaler()
    Xtr_sc  = scaler.fit_transform(X_train)
    Xte_sc  = scaler.transform(X_test)

    model   = LinearRegression()
    model.fit(Xtr_sc, y_train)
    y_pred  = model.predict(Xte_sc)
    metrics = evaluate(y_test, y_pred)
    print_metrics("Linear Regression", metrics)
    return model, scaler, y_pred, metrics


def train_random_forest(X_train, y_train, X_test, y_test,
                        n_estimators: int = 100, random_state: int = 42):
    model   = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    print_metrics("Random Forest Regressor", metrics)
    return model, y_pred, metrics


def train_decision_tree(X_train, y_train, X_test, y_test,
                        max_depth: int = 10, random_state: int = 42):
    model   = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    print_metrics("Decision Tree Regressor", metrics)
    return model, y_pred, metrics


# ── Master trainer ────────────────────────────────────────────────────────────

def train_all_models(X_train, X_test, y_train, y_test):
    """
    Train all three models and return a unified results dict.

    Returns
    -------
    results : dict  {model_name: {"model": ..., "y_pred": ..., "metrics": ...}}
    scaler  : StandardScaler  (fitted for LinearRegression only)
    """
    results = {}

    lr_model, lr_scaler, lr_pred, lr_metrics = train_linear_regression(
        X_train, y_train, X_test, y_test
    )
    results["Linear Regression"] = {
        "model":   lr_model,
        "scaler":  lr_scaler,
        "y_pred":  lr_pred,
        "metrics": lr_metrics,
    }

    rf_model, rf_pred, rf_metrics = train_random_forest(
        X_train, y_train, X_test, y_test
    )
    results["Random Forest"] = {
        "model":   rf_model,
        "scaler":  None,
        "y_pred":  rf_pred,
        "metrics": rf_metrics,
    }

    dt_model, dt_pred, dt_metrics = train_decision_tree(
        X_train, y_train, X_test, y_test
    )
    results["Decision Tree"] = {
        "model":   dt_model,
        "scaler":  None,
        "y_pred":  dt_pred,
        "metrics": dt_metrics,
    }

    return results


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_collection import fetch_stock_data
    from preprocessing   import preprocess

    df = fetch_stock_data("AAPL", "5y")
    X_train, X_test, y_train, y_test, train_df, test_df = preprocess(df)
    results = train_all_models(X_train, X_test, y_train, y_test)
