print("🔥 AN MAIN RUNNING")

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from data import get_stock_data, get_stock_info
from ai import analyze_stock_ai
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd

print("🔥 CLEAN MAIN RUNNING")

dt_model = None
rf_model = None
lr_model = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




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
    global dt_model, rf_model, lr_model

    df = get_stock_data("AAPL")
    df = prepare_features(df)

    X = df[["RSI", "MA20", "MA50", "momentum", "volatility"]]
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    y = df["target"]

    X = X[:-1]
    y = y[:-1]

    dt_model = DecisionTreeClassifier(max_depth=5)
    rf_model = RandomForestClassifier(n_estimators=50)
    lr_model = LogisticRegression()

    dt_model.fit(X, y)
    rf_model.fit(X, y)
    lr_model.fit(X, y)

    print("ALL MODELS TRAINED")


@app.get("/search")
def search_stock(query: str):
    data = [
    # US BIG TECH
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "MSFT", "name": "Microsoft"},
    {"symbol": "NVDA", "name": "Nvidia"},
    {"symbol": "META", "name": "Meta"},
    {"symbol": "AMZN", "name": "Amazon"},
    {"symbol": "GOOGL", "name": "Google"},
    {"symbol": "NFLX", "name": "Netflix"},
    {"symbol": "AMD", "name": "AMD"},
    {"symbol": "INTC", "name": "Intel"},
    {"symbol": "ORCL", "name": "Oracle"},
    {"symbol": "ADBE", "name": "Adobe"},
    {"symbol": "CRM", "name": "Salesforce"},
    {"symbol": "PYPL", "name": "PayPal"},
    {"symbol": "UBER", "name": "Uber"},
    {"symbol": "LYFT", "name": "Lyft"},

    # US OTHERS
    {"symbol": "BA", "name": "Boeing"},
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "DIS", "name": "Disney"},
    {"symbol": "NKE", "name": "Nike"},
    {"symbol": "PFE", "name": "Pfizer"},
    {"symbol": "KO", "name": "Coca Cola"},
    {"symbol": "PEP", "name": "Pepsi"},
    {"symbol": "MCD", "name": "McDonalds"},

    # INDIAN STOCKS
    {"symbol": "RELIANCE.NS", "name": "Reliance"},
    {"symbol": "TCS.NS", "name": "TCS"},
    {"symbol": "INFY.NS", "name": "Infosys"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"symbol": "SBIN.NS", "name": "SBI"},
    {"symbol": "LT.NS", "name": "L&T"},
    {"symbol": "ITC.NS", "name": "ITC"},
    {"symbol": "WIPRO.NS", "name": "Wipro"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
    {"symbol": "MARUTI.NS", "name": "Maruti"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
    {"symbol": "ADANIENT.NS", "name": "Adani Ent"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports"},
    {"symbol": "HCLTECH.NS", "name": "HCL Tech"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement"},
    {"symbol": "ONGC.NS", "name": "ONGC"},
    {"symbol": "NTPC.NS", "name": "NTPC"},
    {"symbol": "POWERGRID.NS", "name": "PowerGrid"},
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
    if dt_model is None or rf_model is None or lr_model is None:
        return {"error": "Models not trained yet"}

    features = latest[["RSI", "MA20", "MA50", "momentum", "volatility"]].values.reshape(1, -1)
    dt_pred = dt_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    lr_pred = lr_model.predict(features)[0]

    dt_prob = dt_model.predict_proba(features)[0][1]
    rf_prob = rf_model.predict_proba(features)[0][1]
    lr_prob = lr_model.predict_proba(features)[0][1]
    votes = dt_pred + rf_pred + lr_pred

    if votes >= 2:
        prediction = "BUY"
    else:
        prediction = "SELL"

    confidence = round(((dt_prob + rf_prob + lr_prob) / 3) * 100, 2)

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
    # ===== FINAL DECISION (HYBRID) =====
    if votes >= 2:
        prediction = "BUY"
    elif votes <= 1:
        prediction = "SELL"
    else:
        prediction = "HOLD"

    # ===== SUPPORT / RESISTANCE =====
    support = float(df["Low"].rolling(20).min().iloc[-1])
    resistance = float(df["High"].rolling(20).max().iloc[-1])

   # ===== TRADE PLAN (FIXED CLEAN LOGIC) =====
    if prediction == "BUY":
        entry = round(price, 2)
        stop_loss = round(min(support, price * 0.98), 2)
        target = round(max(resistance, price * 1.02), 2)

    elif prediction == "SELL":
        entry = round(price, 2)
        stop_loss = round(max(resistance, price * 1.02), 2)
        target = round(min(support, price * 0.98), 2)

    else:
        entry = "-"
        stop_loss = "-"
        target = "-"

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

    # ===== RISK REWARD =====
    if prediction != "HOLD":
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        rr = round(reward / risk, 2) if risk != 0 else 0
    else:
        rr = "-"

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
        "ai_analysis": ai_analysis,
        "models": {
            "decision_tree": "BUY" if dt_pred == 1 else "SELL",
            "random_forest": "BUY" if rf_pred == 1 else "SELL",
            "logistic": "BUY" if lr_pred == 1 else "SELL"
        },
        "votes": int(votes),
        "score": int(score),
    }
# =========================
# TOP OPPORTUNITY
# =========================
@app.get("/top-opportunity")
def top_opportunity():
    symbols = [
        # US
        "AAPL","TSLA","MSFT","GOOGL","AMZN","META","NVDA","AMD","NFLX","INTC",
        "BABA","ORCL","UBER","DIS","PYPL","CRM","ADBE","CSCO","QCOM","SHOP",

        # INDIA
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "SBIN.NS","ITC.NS","LT.NS","WIPRO.NS","AXISBANK.NS",
        "BAJFINANCE.NS","MARUTI.NS","TITAN.NS","ADANIENT.NS","ONGC.NS"
    ]

    best_stock = None
    best_conf = 0

    for s in symbols:
        try:
            res = predict(s)

            if "confidence" in res and isinstance(res["confidence"], (int, float)):
                if res["confidence"] > best_conf:
                    best_conf = res["confidence"]
                    best_stock = res

        except Exception as e:
            continue

    if not best_stock:
        return {"error": "No opportunity found"}

    return best_stock


@app.get("/backtest")
def backtest(symbol: str = "AAPL"):
    df = get_stock_data(symbol)

    if df.empty:
        return {"error": "Invalid symbol"}

    df = prepare_features(df)

    if len(df) < 100:
        return {"error": "Not enough data"}

    correct = 0
    total = 0

    wins = 0
    losses = 0

    for i in range(50, len(df) - 1):
        row = df.iloc[i].copy()

        features = row[["RSI", "MA20", "MA50", "momentum", "volatility"]].values.reshape(1, -1)

        dt_pred = dt_model.predict(features)[0]
        rf_pred = rf_model.predict(features)[0]
        lr_pred = lr_model.predict(features)[0]

        votes = dt_pred + rf_pred + lr_pred
        prediction = 1 if votes >= 2 else 0

        next_price = float(df.iloc[i + 1]["Close"])
        current_price = float(row["Close"])

        actual = 1 if next_price > current_price else 0

        if prediction == actual:
            correct += 1
            wins += 1
        else:
            losses += 1

        total += 1

    accuracy = round((correct / total) * 100, 2) if total else 0

    return {
        "symbol": symbol,
        "accuracy": accuracy,
        "total_trades": total,
        "wins": wins,
        "losses": losses
    }

@app.get("/model-stats")
def model_stats(symbol: str = "AAPL"):
    df = get_stock_data(symbol)

    if df.empty:
        return {"error": "Invalid symbol"}

    df = prepare_features(df)

    if len(df) < 100:
        return {"error": "Not enough data"}

    from sklearn.metrics import confusion_matrix, accuracy_score

    y_true = []
    dt_preds = []
    rf_preds = []
    lr_preds = []

    for i in range(50, len(df) - 1):
        row = df.iloc[i].copy()

        features = row[["RSI", "MA20", "MA50", "momentum", "volatility"]].values.reshape(1, -1)

        dt = dt_model.predict(features)[0]
        rf = rf_model.predict(features)[0]
        lr = lr_model.predict(features)[0]

        next_price = float(df.iloc[i + 1]["Close"])
        current_price = float(row["Close"])

        actual = 1 if next_price > current_price else 0

        y_true.append(actual)
        dt_preds.append(dt)
        rf_preds.append(rf)
        lr_preds.append(lr)

    return {
        "decision_tree_accuracy": round(accuracy_score(y_true, dt_preds) * 100, 2),
        "random_forest_accuracy": round(accuracy_score(y_true, rf_preds) * 100, 2),
        "logistic_accuracy": round(accuracy_score(y_true, lr_preds) * 100, 2),

        "decision_tree_cm": confusion_matrix(y_true, dt_preds).tolist(),
        "random_forest_cm": confusion_matrix(y_true, rf_preds).tolist(),
        "logistic_cm": confusion_matrix(y_true, lr_preds).tolist()
    }