"""
visualization.py
----------------
All chart-generation functions used by both the Streamlit app
and the screenshot-generation script (generate_screenshots.py).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0D1117",
    "panel":   "#161B22",
    "border":  "#30363D",
    "text":    "#E6EDF3",
    "muted":   "#8B949E",
    "blue":    "#58A6FF",
    "green":   "#3FB950",
    "orange":  "#F78166",
    "purple":  "#BC8CFF",
    "yellow":  "#E3B341",
}

MODEL_COLORS = {
    "Linear Regression": PALETTE["blue"],
    "Random Forest":     PALETTE["green"],
    "Decision Tree":     PALETTE["orange"],
}

def _dark_style():
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["panel"],
        "axes.edgecolor":    PALETTE["border"],
        "axes.labelcolor":   PALETTE["text"],
        "xtick.color":       PALETTE["muted"],
        "ytick.color":       PALETTE["muted"],
        "text.color":        PALETTE["text"],
        "grid.color":        PALETTE["border"],
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "legend.facecolor":  PALETTE["panel"],
        "legend.edgecolor":  PALETTE["border"],
        "font.family":       "DejaVu Sans",
    })


# ── 1. Stock price trend ──────────────────────────────────────────────────────

def plot_stock_trend(df: pd.DataFrame, ticker: str = "AAPL",
                     save_path: str = "screenshots/stock_trend.png") -> None:
    _dark_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                             gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"{ticker}  —  Historical Price & Volume",
                 fontsize=16, fontweight="bold", color=PALETTE["text"], y=0.98)

    ax1 = axes[0]
    ax1.plot(df.index, df["Close"],  color=PALETTE["blue"],   lw=1.8, label="Close")
    ax1.plot(df.index, df["MA_20"],  color=PALETTE["yellow"], lw=1.2, ls="--", label="MA 20")
    ax1.plot(df.index, df["MA_50"],  color=PALETTE["purple"], lw=1.2, ls="--", label="MA 50")
    ax1.fill_between(df.index, df["Low"], df["High"],
                     color=PALETTE["blue"], alpha=0.08, label="High–Low range")
    ax1.set_ylabel("Price (USD)", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.4)

    ax2 = axes[1]
    colors = [PALETTE["green"] if c >= o else PALETTE["orange"]
              for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, df["Volume"], color=colors, width=1.5, alpha=0.7)
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    _save(fig, save_path)


# ── 2. Actual vs Predicted ────────────────────────────────────────────────────

def plot_predictions(test_df: pd.DataFrame, y_test: np.ndarray,
                     results: dict, ticker: str = "AAPL",
                     save_path: str = "screenshots/prediction_graph.png") -> None:
    _dark_style()
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"{ticker}  —  Actual vs Predicted Closing Price",
                 fontsize=16, fontweight="bold", color=PALETTE["text"])

    for ax, (name, res) in zip(axes, results.items()):
        color = MODEL_COLORS[name]
        ax.plot(test_df.index, y_test, color=PALETTE["muted"],
                lw=1.5, label="Actual", alpha=0.9)
        ax.plot(test_df.index, res["y_pred"], color=color,
                lw=1.8, label=f"Predicted ({name})", alpha=0.9)
        ax.fill_between(test_df.index, y_test, res["y_pred"],
                        color=color, alpha=0.12)
        m = res["metrics"]
        ax.set_title(
            f"{name}   |   MAE={m['MAE']:.2f}   RMSE={m['RMSE']:.2f}   R²={m['R2']:.4f}",
            fontsize=10, color=PALETTE["muted"])
        ax.set_ylabel("Price (USD)", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    _save(fig, save_path)


# ── 3. Model comparison bar chart ─────────────────────────────────────────────

def plot_model_comparison(results: dict,
                           save_path: str = "screenshots/model_comparison.png") -> None:
    _dark_style()
    metrics_list = ["MAE", "RMSE", "R2"]
    labels = list(results.keys())
    x      = np.arange(len(labels))
    width  = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Model Performance Comparison",
                 fontsize=16, fontweight="bold", color=PALETTE["text"])

    colors = list(MODEL_COLORS.values())

    for idx, (ax, metric) in enumerate(zip(axes, metrics_list)):
        vals = [results[name]["metrics"][metric] for name in labels]
        bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor=PALETTE["border"],
                      linewidth=0.8)
        ax.set_title(metric, fontsize=13, fontweight="bold", color=PALETTE["text"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.4)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, color=PALETTE["text"])

        if metric == "R2":
            ax.set_ylim(0, 1.12)

    plt.tight_layout()
    _save(fig, save_path)


# ── 4. Dashboard mockup ───────────────────────────────────────────────────────

def plot_dashboard_preview(df: pd.DataFrame, test_df: pd.DataFrame,
                           y_test: np.ndarray, results: dict,
                           ticker: str = "AAPL",
                           save_path: str = "screenshots/dashboard.png") -> None:
    _dark_style()
    fig = plt.figure(figsize=(18, 10), facecolor=PALETTE["bg"])
    fig.suptitle(f"AI Stock Market Prediction System  —  {ticker}",
                 fontsize=18, fontweight="bold", color=PALETTE["text"], y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # Top-left: price trend
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df.index, df["Close"], color=PALETTE["blue"], lw=1.5, label="Close")
    ax1.plot(df.index, df["MA_20"], color=PALETTE["yellow"], lw=1, ls="--", label="MA20")
    ax1.plot(df.index, df["MA_50"], color=PALETTE["purple"], lw=1, ls="--", label="MA50")
    ax1.set_title("Historical Close Price with Moving Averages", color=PALETTE["text"], fontsize=11)
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.35)

    # Top-right: RSI
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(df.index, df["RSI"], color=PALETTE["orange"], lw=1.2)
    ax2.axhline(70, color=PALETTE["orange"], ls="--", lw=0.8, alpha=0.7)
    ax2.axhline(30, color=PALETTE["green"],  ls="--", lw=0.8, alpha=0.7)
    ax2.fill_between(df.index, df["RSI"], 70,
                     where=(df["RSI"] >= 70), color=PALETTE["orange"], alpha=0.15)
    ax2.fill_between(df.index, df["RSI"], 30,
                     where=(df["RSI"] <= 30), color=PALETTE["green"],  alpha=0.15)
    ax2.set_title("RSI (14)", color=PALETTE["text"], fontsize=11)
    ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.35)

    # Bottom-left: prediction overlay (best model = RF)
    best = "Random Forest"
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(test_df.index, y_test, color=PALETTE["muted"], lw=1.5, label="Actual")
    ax3.plot(test_df.index, results[best]["y_pred"],
             color=MODEL_COLORS[best], lw=1.8, label=f"Predicted ({best})")
    ax3.set_title("Actual vs Predicted (Random Forest)", color=PALETTE["text"], fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.35)

    # Bottom-right: R² comparison
    ax4 = fig.add_subplot(gs[1, 2])
    names  = list(results.keys())
    r2vals = [results[n]["metrics"]["R2"] for n in names]
    bars   = ax4.barh(names, r2vals, color=list(MODEL_COLORS.values()),
                      edgecolor=PALETTE["border"])
    ax4.set_xlim(0, 1.1)
    for bar, val in zip(bars, r2vals):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9, color=PALETTE["text"])
    ax4.set_title("R² Score Comparison", color=PALETTE["text"], fontsize=11)
    ax4.grid(axis="x", alpha=0.35)

    _save(fig, save_path)


# ── 5. Architecture diagram ───────────────────────────────────────────────────

def plot_architecture(save_path: str = "screenshots/architecture.png") -> None:
    _dark_style()
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=PALETTE["bg"])
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])
    fig.suptitle("System Architecture  —  AI Stock Market Prediction",
                 fontsize=15, fontweight="bold", color=PALETTE["text"])

    stages = [
        ("Data\nCollection",   1.0,  3.5, PALETTE["blue"],   "yfinance\nYahoo Finance"),
        ("Preprocessing\n& EDA",3.5,  3.5, PALETTE["purple"], "Pandas / NumPy\nCleaning & Features"),
        ("Model\nTraining",     6.5,  3.5, PALETTE["yellow"], "Linear Reg\nRandom Forest\nDecision Tree"),
        ("Prediction\n& Metrics",9.5, 3.5, PALETTE["orange"], "MAE / RMSE\nR² Score"),
        ("Streamlit\nDashboard",12.5, 3.5, PALETTE["green"],  "Interactive UI\nPlotly Charts"),
    ]

    for i, (title, x, y, color, sub) in enumerate(stages):
        fancy = mpatches.FancyBboxPatch(
            (x - 1.0, y - 1.4), 2.0, 2.8,
            boxstyle="round,pad=0.12",
            facecolor=PALETTE["panel"], edgecolor=color, linewidth=2.5)
        ax.add_patch(fancy)
        ax.text(x, y + 0.7, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x, y - 0.45, sub,  ha="center", va="center",
                fontsize=8, color=PALETTE["muted"])

        if i < len(stages) - 1:
            nx = stages[i + 1][1]
            ax.annotate("",
                xy=(nx - 1.05, y), xytext=(x + 1.05, y),
                arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=1.8))

    ax.text(8, 0.45,
            "Modular Python Pipeline  •  Scikit-learn  •  Streamlit  •  Matplotlib / Plotly",
            ha="center", fontsize=9, color=PALETTE["muted"], style="italic")

    _save(fig, save_path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Saved → {path}")
