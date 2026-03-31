print("🔥 CLEAN MAIN RUNNING")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from data import get_stock_data, get_stock_info
from ai import analyze_stock_ai

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def prepare_features(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["volatility"] = df["Close"].pct_change().rolling(10).std()
    df["momentum"] = df["Close"].pct_change(10)

    df.dropna(inplace=True)
    return df


@app.on_event("startup")
def train_model():
    global model

    df = get_stock_data("AAPL")
    df = prepare_features(df)

    X = df[["RSI", "MA20", "MA50", "momentum", "volatility"]]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    y = df["target"]

    X = X[:-1]
    y = y[:-1]

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)

    print("✅ MODEL TRAINED")


@app.get("/search")
def search_stock(query: str):
    data = [
        {"symbol": "AAPL", "name": "Apple"},
        {"symbol": "TSLA", "name": "Tesla"},
        {"symbol": "TXN", "name": "Texas Instruments"},
    ]

    return [s for s in data if query.lower() in s["symbol"].lower() or query.lower() in s["name"].lower()]


print("🔥 CLEAN MAIN RUNNING")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from data import get_stock_data, get_stock_info
from ai import analyze_stock_ai

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def prepare_features(df):
    df = df.copy()

    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["volatility"] = df["Close"].pct_change().rolling(10).std()
    df["momentum"] = df["Close"].pct_change(10)

    df.dropna(inplace=True)
    return df


@app.on_event("startup")
def train_model():
    global model

    df = get_stock_data("AAPL")
    df = prepare_features(df)

    X = df[["RSI", "MA20", "MA50", "momentum", "volatility"]]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    y = df["target"]

    X = X[:-1]
    y = y[:-1]

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X, y)

    print("✅ MODEL TRAINED")


@app.get("/search")
def search_stock(query: str):
    data = [
        {"symbol": "AAPL", "name": "Apple"},
        {"symbol": "TSLA", "name": "Tesla"},
        {"symbol": "TXN", "name": "Texas Instruments"},
    ]

    return [s for s in data if query.lower() in s["symbol"].lower() or query.lower() in s["name"].lower()]


@app.get("/predict")
def predict(symbol: str = Query("AAPL")):
    df = get_stock_data(symbol)

    if df.empty:
        return {"error": "Invalid symbol"}

    df = prepare_features(df)

    if df.empty:
        return {"error": "Not enough data"}

    latest = df.iloc[-1]

    # ===== PRICE =====
    price = float(latest["Close"])

    # ===== ML =====
    features = latest[["RSI", "MA20", "MA50", "momentum", "volatility"]].values.reshape(1, -1)
    prob = model.predict_proba(features)[0][1]
    confidence = round(prob * 100, 2)

    # ===== SIGNAL ENGINE =====
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    rsi = float(latest["RSI"])
    momentum = float(latest["momentum"])

    signals = []
    score = 0

    if ma20 > ma50:
        signals.append("MA20 > MA50 (Bullish)")
        score += 1
    else:
        signals.append("MA20 < MA50 (Bearish)")
        score -= 1

    if rsi < 30:
        signals.append("RSI Oversold")
        score += 2
    elif rsi > 70:
        signals.append("RSI Overbought")
        score -= 2
    else:
        signals.append("RSI Neutral")

    if momentum > 0:
        signals.append("Momentum Positive")
        score += 1
    else:
        signals.append("Momentum Negative")
        score -= 1

    # ===== FINAL DECISION =====
    if score >= 2:
        prediction = "BUY"
    elif score <= -2:
        prediction = "SELL"
    else:
        prediction = "HOLD"

    # ===== SUPPORT / RESISTANCE =====
    support = float(df["Low"].rolling(20).min().iloc[-1])
    resistance = float(df["High"].rolling(20).max().iloc[-1])

    # ===== TRADE PLAN =====
    if prediction == "BUY":
        entry = round(price, 2)
        stop_loss = round(support, 2)
        target = round(resistance, 2)
    elif prediction == "SELL":
        entry = round(price, 2)
        stop_loss = round(resistance, 2)
        target = round(support, 2)
    else:
        entry = "-"
        stop_loss = "-"
        target = "-"

    if prediction != "HOLD":
        risk = float(abs(entry - stop_loss))
        reward = float(abs(target - entry))
        rr = round(reward / risk, 2) if risk != 0 else 0
    else:
        rr = "-"

    # ===== AI =====
    trend = "Uptrend" if ma20 > ma50 else "Downtrend"

    ai_analysis = analyze_stock_ai(
        symbol,
        trend,
        round(rsi, 2),
        prediction,
        confidence
    )

    # ===== COMPANY INFO =====
    info = get_stock_info(symbol)

    return {
        "symbol": symbol,
        "company": info.get("name", symbol),
        "price": round(price, 2),
        "prediction": prediction,
        "confidence": confidence,
        "signals": signals,
        "trade_plan": {
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": rr
        },
        "ai_analysis": ai_analysis
    }