# 🚀 AI Stock Intelligence System

A **quant-inspired AI trading decision system** that combines machine learning, technical analysis, and (upcoming) sentiment intelligence to generate **high-confidence trading signals**.

---

## 🧠 Core Idea

This is NOT a simple price prediction model.

The system is designed to:
- Filter low-quality trades
- Combine multiple signals (ML + technicals)
- Generate structured trade plans
- Simulate real trading performance

---

## ⚙️ System Architecture

### 1. Data Layer
- Historical stock data (OHLCV)
- Real-time fetching via yfinance

### 2. Feature Engineering
- RSI (Relative Strength Index)
- Moving Averages (MA20, MA50)
- Momentum
- Volatility

### 3. Model Layer
- Logistic Regression
- Random Forest
- (In Progress) XGBoost → primary model

### 4. Decision Engine
- Multi-model voting
- Confidence scoring
- Signal confluence (trend + momentum + RSI)

### 5. Execution Layer
- Trade plan generation:
  - Entry
  - Stop-loss
  - Target
- Risk/Reward calculation

### 6. Evaluation Layer
- Backtesting engine
- Accuracy tracking
- Profit simulation

---

## 🔥 Features

- 📊 AI-based BUY / SELL / HOLD signals
- 🎯 Confidence scoring (multi-model weighted)
- 📈 Technical signal analysis (RSI, trend, momentum)
- 💼 Trade planning with risk management
- 🔍 Multi-stock opportunity scanner
- ⭐ Watchlist tracking
- 🧪 Backtesting with performance metrics

---

## 🧪 Upcoming Upgrades (In Progress)

- ⚡ XGBoost integration (primary model)
- 🧠 Market regime detection (trend vs sideways)
- 📰 Sentiment analysis microservice (news-based)
- 🔐 Trust-weighted news scoring
- 📉 Advanced metrics (Sharpe ratio, drawdown)

---

## 🛠 Tech Stack

- Backend: FastAPI
- Frontend: React
- ML: scikit-learn, XGBoost (planned)
- Data: pandas, yfinance

---

## 🚀 How to Run

### Backend

```bash
pip install -r requirements.txt
uvicorn main:app --reload
