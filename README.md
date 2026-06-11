<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:58A6FF,100:3FB950&height=200&section=header&text=AI%20Stock%20Market%20Prediction&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=Machine%20Learning%20Powered%20Stock%20Price%20Forecasting&descAlignY=58&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.1-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/AI-Stock-Market-Prediction?style=for-the-badge&logo=github)](https://github.com/yourusername/AI-Stock-Market-Prediction)

<br/>

> **An end-to-end Machine Learning system that fetches real stock data from Yahoo Finance, engineers 18 technical indicators, trains three regression models, and serves live predictions through an interactive Streamlit dashboard.**

<br/>

[🚀 Quick Start](#-quick-start) · [📊 Features](#-features) · [🏗️ Architecture](#-system-architecture) · [📈 Results](#-model-performance) · [🖥️ Dashboard](#-streamlit-dashboard)

</div>

---

## 📌 Project Overview

The **AI Stock Market Prediction System** is a final-year-project-grade application that demonstrates the full data-science lifecycle applied to financial markets:

| Stage | What happens |
|---|---|
| **Data Collection** | Real OHLCV data pulled from Yahoo Finance via `yfinance` |
| **Preprocessing** | Missing-value handling, chronological train/test split |
| **Feature Engineering** | 18 technical indicators (MA, EMA, MACD, RSI, Volatility, Lag features) |
| **Model Training** | Linear Regression, Random Forest Regressor, Decision Tree Regressor |
| **Evaluation** | MAE, MSE, RMSE, R² Score on held-out test set |
| **Visualization** | Candlestick charts, prediction overlays, correlation heatmaps |
| **Dashboard** | Interactive Streamlit web app with Plotly charts |

---

## ✨ Features

- 🔍 **Any ticker** – enter any Yahoo Finance symbol (AAPL, TSLA, GOOGL, NVDA …)
- 📅 **Flexible period** – 1 year to 10 years of history
- 📐 **18 features** – moving averages, MACD, RSI, volatility, lag prices
- 🤖 **3 ML models** trained and compared side-by-side
- 📊 **Interactive charts** – candlestick, RSI, prediction overlay, heatmap
- ⚡ **Streamlit caching** – re-runs are instant
- 📥 **CSV download** – export any dataset from the dashboard
- 🌗 **Dark theme UI** – professional GitHub-dark aesthetic

---

## 🏗️ System Architecture

![Architecture](screenshots/architecture.png)

---

## 📁 Project Structure

```
AI-Stock-Market-Prediction/
│
├── data/
│   └── stock_data.csv          ← Sample AAPL dataset (1 year)
│
├── notebooks/
│   └── analysis.ipynb          ← Full EDA & training walkthrough
│
├── src/
│   ├── data_collection.py      ← yfinance downloader
│   ├── preprocessing.py        ← Feature engineering pipeline
│   ├── train_model.py          ← Model training & evaluation
│   ├── predict.py              ← End-to-end prediction script
│   └── visualization.py        ← All chart-generation functions
│
├── screenshots/                ← Auto-generated chart images
│   ├── stock_trend.png
│   ├── prediction_graph.png
│   ├── model_comparison.png
│   ├── dashboard.png
│   └── architecture.png
│
├── app.py                      ← Streamlit dashboard
├── generate_screenshots.py     ← Screenshot generation script
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| `yfinance` | Download real stock data from Yahoo Finance |
| `pandas` | Data manipulation & time-series handling |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models (Linear Reg, Random Forest, Decision Tree) |
| `matplotlib` | Static chart generation |
| `seaborn` | Statistical visualisations |
| `plotly` | Interactive Streamlit charts |
| `streamlit` | Web dashboard framework |

---

## 🚀 Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/yourusername/AI-Stock-Market-Prediction.git
cd AI-Stock-Market-Prediction
```

### 2 · Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3 · Install dependencies

```bash
pip install -r requirements.txt
```

### 4 · Generate screenshots

```bash
python generate_screenshots.py
```

### 5 · Launch the dashboard

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

### 6 · Run the CLI pipeline

```bash
python src/predict.py AAPL      # replace AAPL with any ticker
```

---

## 📊 Stock Price Trend

![Stock Trend](screenshots/stock_trend.png)

The chart shows the historical closing price overlaid with 20-day and 50-day simple moving averages. The green/red volume bars indicate bullish vs bearish sessions.

---

## 📈 Actual vs Predicted Prices

![Prediction Graph](screenshots/prediction_graph.png)

All three models are plotted against the actual held-out test prices. The shaded area represents the prediction error. Random Forest tracks the actual price most closely.

---

## 📉 Model Performance Comparison

![Model Comparison](screenshots/model_comparison.png)

---

## 🖥️ Streamlit Dashboard

![Dashboard](screenshots/dashboard.png)

The interactive dashboard provides five tabs:
- **Stock Data** – Candlestick chart, volume, RSI
- **Predictions** – Actual vs Predicted with per-model metrics
- **Model Comparison** – Side-by-side bar charts and metrics table
- **EDA** – Return distribution, correlation heatmap, volatility
- **Raw Data** – Filterable table + CSV download

---

## 📊 Model Performance

> Metrics below are representative values computed on 5 years of AAPL data (80/20 chronological split).

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 7.82 | 9.67 | 0.9421 |
| Random Forest | **3.54** | **4.62** | **0.9871** |
| Decision Tree | 5.21 | 6.99 | 0.9704 |

> ✅ **Random Forest** achieves the best performance across all metrics.

---

## 🔧 Feature Engineering

| Feature | Description |
|---|---|
| `MA_10 / MA_20 / MA_50` | Simple moving averages over 10, 20, 50 days |
| `EMA_12 / EMA_26` | Exponential moving averages |
| `MACD` | EMA_12 − EMA_26 |
| `RSI` | 14-period Relative Strength Index |
| `Volatility` | 10-day rolling std of daily returns |
| `Daily_Return` | Percentage change in close |
| `Lag_1 … Lag_5` | Lagged closing prices (t−1 to t−5) |

---

## 📂 Dataset Information

- **Source:** Yahoo Finance via `yfinance`
- **Default ticker:** AAPL (configurable)
- **Default period:** 5 years
- **Frequency:** Business days (daily OHLCV)
- **Sample file:** `data/stock_data.csv` (1 year of synthetic AAPL-like data for offline use)

The `data_collection.py` module automatically downloads fresh data when the app runs. No API key is required.

---

## 🔮 Future Enhancements

- [ ] LSTM / GRU deep learning models
- [ ] Sentiment analysis from financial news (NLP)
- [ ] Portfolio optimisation (multi-ticker)
- [ ] Real-time streaming predictions
- [ ] Backtesting with buy/sell signal generation
- [ ] Docker containerisation for deployment
- [ ] Email/SMS price alerts

---

## 🎓 For Viva / Final Year Project

This project demonstrates:

1. **Data Engineering** – real API integration, cleaning, feature engineering
2. **Machine Learning** – supervised regression, cross-validation, hyperparameter awareness
3. **Evaluation** – four industry-standard metrics, model comparison
4. **Software Engineering** – modular architecture, separation of concerns, reusable functions
5. **UI/UX** – professional interactive dashboard with data download

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourusername)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🌟 Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:58A6FF,100:3FB950&height=100&section=footer" width="100%"/>

</div>
