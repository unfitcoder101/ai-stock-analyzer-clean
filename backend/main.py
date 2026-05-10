import joblib
import numpy as np
import random
import pandas as pd
import requests
import csv
import os
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from data import get_stock_data, get_stock_info
from ai import analyze_stock_ai
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier 
import shap


print("🔥 CLEAN MAIN RUNNING")

dt_model = None
rf_model = None
lr_model = None
xgb_model = None 
scaler = None
explainer = None   # ✅ ADD



model_weights = {
    "dt": 0.25,
    "rf": 0.25,
    "lr": 0.25,
    "xgb": 0.25
}
last_trade_time = None

paper_trades = []
paper_balance = 10000
open_positions = []
# ✅ ADD MEMORY
model_performance = {
    "dt": [],
    "rf": [],
    "lr": [],
    "xgb": []
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def log_experiment(version, change, results):
    file_exists = os.path.isfile("experiments.csv")

    with open("experiments.csv", mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "version",
                "change",
                "return_pct",
                "sharpe",
                "win_rate",
                "drawdown"
            ])

        writer.writerow([
            datetime.now(),
            version,
            change,
            results.get("return_pct"),
            results.get("sharpe_ratio"),
            results.get("win_rate"),
            results.get("max_drawdown")
        ])

def prepare_features(df):
    if df is None or df.empty:
        return pd.DataFrame()

    if "Close" not in df.columns:
        return pd.DataFrame()
    
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
    global dt_model, rf_model, lr_model, xgb_model,  scaler

   
    symbols = [
    # US BIG TECH
    "AAPL","TSLA","MSFT","GOOGL","AMZN",

    # FINANCE
    "JPM","GS","BAC","MS",

    # OTHER
    "UBER","DIS","PYPL","CRM","ADBE",

    # INDIA
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"
     ]

    all_data = []

    
    for s in symbols:
        try:
            temp_df = get_stock_data(s)
        except Exception as e:
            print(f"❌ Failed for {s}: {e}")
            continue

    # ✅ FORCE CLEAN SINGLE-LEVEL COLUMNS
        if isinstance(temp_df.columns, pd.MultiIndex):
            temp_df.columns = temp_df.columns.get_level_values(0)

        temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]  # remove duplicates

        temp_df = prepare_features(temp_df)

        if temp_df is None or temp_df.empty:
            continue

        if not temp_df.empty:
            all_data.append(temp_df)

    if not all_data:
        print("🔥 No valid data — skipping training")
        return

    df = pd.concat(all_data, ignore_index=True)
    # ✅ ADD HERE (correct place)
    if not all_data:
        print("🔥 No data fetched — skipping training")
        return
    
    df = df.reset_index(drop=True)

    X = df[["RSI", "MA20", "MA50", "momentum", "volatility"]]
   

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # ✅ FORCE SINGLE CLOSE COLUMN
    
    df["Close"] = df["Close"].astype(float)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int) 

    y = df["target"]

    X = X[:-1]
    y = y[:-1]
    
    dt_model = DecisionTreeClassifier(max_depth=5)
    rf_model = RandomForestClassifier(n_estimators=50)
    lr_model = LogisticRegression()

    # ✅ NEW MODEL
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
       )


    dt_model.fit(X, y)
    rf_model.fit(X, y)
    lr_model.fit(X, y)
    xgb_model.fit(X, y)   # ✅ ADD

# ✅ SHAP EXPLAINER
    global explainer
    explainer = shap.TreeExplainer(xgb_model)

    print("ALL MODELS TRAINED")
    joblib.dump(dt_model, "dt.pkl")
    joblib.dump(rf_model, "rf.pkl")
    joblib.dump(lr_model, "lr.pkl")
    joblib.dump(xgb_model, "xgb.pkl") 
    joblib.dump(scaler, "scaler.pkl")


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
    
    import requests

    global last_trade_time
    global paper_trades, open_positions

# ✅ GET SENTIMENT (LOCAL ONLY)
    try:
        res = requests.get(
        f"http://127.0.0.1:8001/sentiment?symbol={symbol}",
        timeout=2
        )
        sentiment_score = res.json().get("score", 0)
        sentiment_score_norm = round((sentiment_score + 1) * 50, 2)

    except:
        sentiment_score = 0
        sentiment_score_norm = 50   # ✅ VERY IMPORTANT FIX

    df = get_stock_data(symbol)
    # ✅ STEP 3 — FIX MULTI-INDEX (PASTE HERE)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ✅ STEP 4 — CLEAN DATA (PASTE HERE)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(inplace=True)

    if df.empty:
        return {"error": "Invalid symbol"}

    df = prepare_features(df)
    if df is None or df.empty or len(df) < 50:
        return {"error": "Not enough data from API"}

    if df.empty:
        return {"error": "Not enough data"}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    price = float(latest["Close"])
    
    rsi = float(latest["RSI"])
    # =========================
    # BREAKOUT LEVELS
    # =========================
    recent_high = float(df["High"].rolling(20).max().iloc[-2])
    recent_low = float(df["Low"].rolling(20).min().iloc[-2])

    is_breakout_up = price > recent_high
    is_breakout_down = price < recent_low

    # =========================
# FAKE BREAKOUT FILTER
# =========================
    valid_breakout_up = (
    is_breakout_up and
    price > prev["High"] and
    rsi > 50
    )

    valid_breakout_down = (
    is_breakout_down and
    price < prev["Low"] and
    rsi < 50
    )

    # =========================
# REGIME DETECTION
# =========================
    ma20 = float(latest["MA20"])
    ma50 = float(latest["MA50"])
    rsi = float(latest["RSI"])
    volatility = float(latest["volatility"])

    if ma20 > ma50 and volatility > 0.01:
        regime = "TREND_UP"
    elif ma20 < ma50 and volatility > 0.01:
        regime = "TREND_DOWN"
    else:
        regime = "RANGE"

    # ===== PRICE =====

    price = float(latest["Close"])

    # ===== ML =====
    if dt_model is None or rf_model is None or lr_model is None or xgb_model is None:
        return {"error": "Models not trained yet"}
    
    features_df = pd.DataFrame([latest[["RSI", "MA20", "MA50", "momentum", "volatility"]]])
    features = scaler.transform(features_df)
    dt_pred = dt_model.predict(features)[0]
    rf_pred = rf_model.predict(features)[0]
    lr_pred = lr_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]   # ✅ ADD

    # ✅ STORE LAST PREDICTIONS (for future learning system)
    last_preds = {
    "dt": int(dt_pred),
    "rf": int(rf_pred),
    "lr": int(lr_pred),
    "xgb": int(xgb_pred),
    "price": price
    }
    dt_prob = dt_model.predict_proba(features)[0][1]
    rf_prob = rf_model.predict_proba(features)[0][1]
    lr_prob = lr_model.predict_proba(features)[0][1]
    xgb_prob = xgb_model.predict_proba(features)[0][1]

    score_ml = (
    dt_pred * model_weights["dt"] +
    rf_pred * model_weights["rf"] +
    lr_pred * model_weights["lr"] +
    xgb_pred * model_weights["xgb"]
    )

# ✅ SHAP VALUES
    shap_values = explainer.shap_values(features)[0]

    feature_names = ["RSI", "MA20", "MA50", "momentum", "volatility"]

# Convert to readable format
    shap_dict = {
    feature_names[i]: float(shap_values[i])
    for i in range(len(feature_names))
    }

# Sort by importance
    shap_sorted = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)) 

    votes = dt_pred + rf_pred + lr_pred + xgb_pred

# ✅ AGREEMENT LEVEL
    agreement = f"{votes}/4 models bullish"


    ml_confidence = round((
    dt_prob * model_weights["dt"] +
    rf_prob * model_weights["rf"] +
    lr_prob * model_weights["lr"] +
    xgb_prob * model_weights["xgb"]
    ) * 100, 2)

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

    # ✅ NORMALIZE TECH SCORE (VERY IMPORTANT)
    technical_score = round(((score + 4) / 8) * 100, 2)
    # ✅ FINAL COMBINED CONFIDENCE
    # ✅ FINAL MULTI-MODAL CONFIDENCE
    confidence = round(
        0.5 * ml_confidence +
        0.3 * technical_score +
        0.2 * sentiment_score_norm,
        2
    )

    # =========================
# FINAL DECISION ENGINE (BREAKOUT + REGIME)
# =========================

    prediction = "HOLD"

# 🔥 TREND + BREAKOUT (HIGH QUALITY TRADES)
    if regime == "TREND_UP":
        if valid_breakout_up and score_ml >= 0.6:
            prediction = "BUY"

    elif regime == "TREND_DOWN":
        if valid_breakout_down and score_ml <= 0.4:
            prediction = "SELL"

# 🔁 RANGE (MEAN REVERSION)
    elif regime == "RANGE":
        if rsi < 30:
            prediction = "BUY"
        elif rsi > 70:
            prediction = "SELL"

# 🔒 CONFIDENCE FILTER
    if confidence < 55:
        prediction = "HOLD"

# =========================
# VOLATILITY FILTER (ADD HERE)
# =========================
    if volatility < 0.005:
        prediction = "HOLD"

# 🧠 SENTIMENT FILTER
    if prediction == "BUY" and sentiment_score < -0.2:
        prediction = "HOLD"

    if prediction == "SELL" and sentiment_score > 0.2:
        prediction = "HOLD" 

    # =========================
    # TRADE COOLDOWN (ADD HERE)
    # =========================
    global last_trade_time

    if last_trade_time:
        if datetime.now() - last_trade_time < timedelta(minutes=30):
            prediction = "HOLD"
 

    # ===== SUPPORT / RESISTANCE =====
    support = float(df["Low"].rolling(20).min().iloc[-1])
    resistance = float(df["High"].rolling(20).max().iloc[-1])

    # ✅ CAPITAL SETTINGS
    capital = 10000   # you can make this dynamic later
    risk_per_trade = 0.02  # 2% risk per trade

    # =========================
# ATR CALCULATION
# =========================
    df["high_low"] = df["High"] - df["Low"]
    df["high_close"] = abs(df["High"] - df["Close"].shift())
    df["low_close"] = abs(df["Low"] - df["Close"].shift())

    df["tr"] = df[["high_low", "high_close", "low_close"]].max(axis=1)
    df["ATR"] = df["tr"].rolling(14).mean()

    atr = float(df["ATR"].iloc[-1])

   # ===== TRADE PLAN (FIXED CLEAN LOGIC) =====
    if prediction == "BUY":
        entry = price
        stop_loss = round(price - (1.5 * atr), 2)
        target = round(price + (2 * atr), 2)

    elif prediction == "SELL":
        entry = price
        stop_loss = round(price + (1.5 * atr), 2)
        target = round(price - (2 * atr), 2)
    # ✅ POSITION SIZING
    if prediction != "HOLD":
        risk_per_unit = abs(entry - stop_loss)

        if risk_per_unit > 0:
            position_size = (capital * risk_per_trade) / risk_per_unit
            position_size = int(position_size)
        else:
            position_size = 0

        capital_used = round(position_size * entry, 2)
    else:
        entry = "-"
        stop_loss = "-"
        target = "-"
        position_size = 0
        capital_used = 0

    # =========================
# STORE PAPER TRADE
# =========================
    if prediction in ["BUY", "SELL"]:

        last_trade_time = datetime.now()

        trade = {
            "time": str(datetime.now()),
            "symbol": symbol,
            "prediction": prediction,
            "price": round(price, 2),
            "confidence": confidence,
            "regime": regime,
            "target": target,
            "stop_loss": stop_loss,
            "position_size": position_size
        }

        paper_trades.append(trade)
        open_positions.append(trade)

    # ===== AI =====
    trend = "Uptrend" if ma20 > ma50 else "Downtrend"

    reason = []

    if ma20 > ma50:
        reason.append("Bullish trend (MA crossover)")
    else:
        reason.append("Bearish trend (MA crossover)")

    if rsi < 30:
        reason.append("Stock oversold")
    elif rsi > 70:
        reason.append("Stock overbought")

    if momentum > 0:
        reason.append("Positive momentum")
    else:
        reason.append("Negative momentum")

    ai_analysis = analyze_stock_ai(
    symbol,
    trend,
    round(rsi, 2),
    prediction,
    confidence
    )

    ai_analysis += "\n\nReasons:\n" + ", ".join(reason)


    # ===== COMPANY INFO =====
    info = get_stock_info(symbol)

    # ===== RISK REWARD =====
    if prediction != "HOLD":
        risk = abs(entry - stop_loss)
        reward = abs(target - entry)
        rr = round(reward / risk, 2) if risk != 0 else 0
    else:
        rr = "-"

    best_model = max([
        ("Decision Tree", dt_prob),
        ("Random Forest", rf_prob),
        ("Logistic", lr_prob),
        ("XGBoost", xgb_prob)  
    ], key=lambda x: x[1])[0]

    return {
        "symbol": symbol,
        "sentiment_score": sentiment_score,
        "best_model": best_model,
        "company": info.get("name", symbol),
        "price": round(price, 2),
        "prediction": prediction,
        "confidence": confidence,
        "confidence_breakdown": {
        "ml_confidence": ml_confidence,
        "technical_score": technical_score,
        "agreement": agreement
        },
        "signals": signals,
        "trade_plan": {
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": rr,
         # ✅ NEW
            "position_size": position_size,
            "capital_used": capital_used,
            "risk_per_trade_pct": 2
        },
        "ai_analysis": ai_analysis,
        "sentiment": {
        "score_raw": sentiment_score,
        "score_normalized": sentiment_score_norm
        },
        "breakout": {
            "recent_high": recent_high,
            "recent_low": recent_low,
            "is_breakout_up": is_breakout_up,
            "is_breakout_down": is_breakout_down
        },
        "breakout_quality": {
            "valid_up": valid_breakout_up,
            "valid_down": valid_breakout_down
        },
        "models": {
            "decision_tree": "BUY" if dt_pred == 1 else "SELL",
            "random_forest": "BUY" if rf_pred == 1 else "SELL",
            "logistic": "BUY" if lr_pred == 1 else "SELL",
            "xgboost": "BUY" if xgb_pred == 1 else "SELL"  
        },
        "votes": int(votes),
        "regime": regime,
        "score": int(score),
        "explainability": shap_sorted
    }
# =========================
# TOP OPPORTUNITY
# =========================
@app.get("/top-opportunity")
def top_opportunity():
    capital = 10000  # total portfolio capital
    symbols = [
        # US
        "AAPL","TSLA","MSFT","GOOGL","AMZN","META","NVDA","AMD","NFLX","INTC",
        "BABA","ORCL","UBER","DIS","PYPL","CRM","ADBE","CSCO","QCOM","SHOP",

        # INDIA
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "SBIN.NS","ITC.NS","LT.NS","WIPRO.NS","AXISBANK.NS",
        "BAJFINANCE.NS","MARUTI.NS","TITAN.NS","ADANIENT.NS","ONGC.NS"
    ]

    results = []   # ✅ STORE ALL STOCKS

    for s in symbols:
        try:
            res = predict(s)

            if "confidence" in res and isinstance(res["confidence"], (int, float)):
                results.append({
                    "symbol": res["symbol"],
                    "confidence": res["confidence"],
                    "prediction": res["prediction"]
                })

        except:
            continue

    # ✅ SORT ALL STOCKS
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    top_stocks = results[:5]

    # ✅ TOTAL CONFIDENCE
    total_conf = sum([s["confidence"] for s in top_stocks])

    portfolio = []

    for stock in top_stocks:
        if total_conf > 0:
            weight = stock["confidence"] / total_conf
        else:
            weight = 0

        allocation = round(capital * weight, 2)

        portfolio.append({
        "symbol": stock["symbol"],
        "prediction": stock["prediction"],
        "confidence": stock["confidence"],
        "weight": round(weight, 3),
        "capital_allocated": allocation
        })

    
    return {
        "total_capital": capital,
        "portfolio": portfolio
    }

@app.get("/backtest")
def backtest(symbol: str = "AAPL"):
    df = get_stock_data(symbol)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]

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

        features_df = pd.DataFrame([row[["RSI", "MA20", "MA50", "momentum", "volatility"]]])
        features = scaler.transform(features_df)
        dt_pred = dt_model.predict(features)[0]
        rf_pred = rf_model.predict(features)[0]
        lr_pred = lr_model.predict(features)[0]
        xgb_pred = xgb_model.predict(features)[0]  # ✅ FIXED

    # ✅ WEIGHTED ML SCORE
        score_ml = (
        dt_pred * model_weights["dt"] +
        rf_pred * model_weights["rf"] +
        lr_pred * model_weights["lr"] +
        xgb_pred * model_weights["xgb"]
    )

    # ✅ FINAL ML PREDICTION
        prediction = 1 if score_ml >= 0.5 else 0

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

    capital = 10000
    profit = 0

    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        next_price = float(df.iloc[i + 1]["Close"])
        current_price = float(row["Close"])

        features = row[["RSI", "MA20", "MA50", "momentum", "volatility"]].values.reshape(1, -1)

        pred = dt_model.predict(features)[0]

        trade_amount = 1000  # fixed per trade

        if pred == 1:  # BUY
            change = (next_price - current_price) / current_price
        else:  # SELL
            change = (current_price - next_price) / current_price

        pnl = trade_amount * change
        profit += pnl

    final_capital = capital + profit

    return {
    "symbol": symbol,
    "accuracy": round((wins / total) * 100, 2),
    "total_trades": total,
    "wins": wins,
    "losses": losses,
    "profit": round(profit, 2),
    "final_capital": round(final_capital, 2)
    }

@app.get("/simulate-profit")
def simulate_profit(symbol: str = "AAPL"):
    df = get_stock_data(symbol)
    df = prepare_features(df)

    capital = 10000
    position = 0

    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        features_df = pd.DataFrame([row[["RSI", "MA20", "MA50", "momentum", "volatility"]]])
        features = scaler.transform(features_df)
        pred = rf_model.predict(features)[0]

        next_price = float(df.iloc[i + 1]["Close"])
        current_price = float(row["Close"])

        if pred == 1:  # BUY
            profit = next_price - current_price
        else:  # SELL
            profit = current_price - next_price

        capital += profit

    return {
        "final_capital": round(capital, 2),
        "profit": round(capital - 10000, 2)
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

        features_df = pd.DataFrame([row[["RSI", "MA20", "MA50", "momentum", "volatility"]]])
        features = scaler.transform(features_df)
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

    global model_weights

    dt_acc = accuracy_score(y_true, dt_preds)
    rf_acc = accuracy_score(y_true, rf_preds)
    lr_acc = accuracy_score(y_true, lr_preds)

# ✅ ADD XGB
    xgb_preds = []
    for i in range(50, len(df) - 1):
        row = df.iloc[i]
        features = row[["RSI", "MA20", "MA50", "momentum", "volatility"]].values.reshape(1, -1)
        xgb = xgb_model.predict(features)[0]
        xgb_preds.append(xgb)

    xgb_acc = accuracy_score(y_true, xgb_preds)

    # ✅ NORMALIZE → WEIGHTS
    total = dt_acc + rf_acc + lr_acc + xgb_acc

    model_weights = {
        "dt": dt_acc / total,
        "rf": rf_acc / total,
        "lr": lr_acc / total,
        "xgb": xgb_acc / total
    }

    return {
    "decision_tree_accuracy": round(dt_acc * 100, 2),
    "random_forest_accuracy": round(rf_acc * 100, 2),
    "logistic_accuracy": round(lr_acc * 100, 2),
    "xgboost_accuracy": round(xgb_acc * 100, 2),   # ✅ ADD

    "weights": model_weights,  # ✅ IMPORTANT

    "decision_tree_cm": confusion_matrix(y_true, dt_preds).tolist(),
    "random_forest_cm": confusion_matrix(y_true, rf_preds).tolist(),
    "logistic_cm": confusion_matrix(y_true, lr_preds).tolist()
    }

def update_model_performance(symbol, last_preds, next_price):
    global model_performance, model_weights

    actual = 1 if next_price > last_preds["price"] else 0

    for model in ["dt", "rf", "lr", "xgb"]:
        pred = last_preds[model]

        if pred == actual:
            model_performance[model].append(1)
        else:
            model_performance[model].append(0)

        # keep only last 50 trades
        model_performance[model] = model_performance[model][-50:]

    # ✅ UPDATE WEIGHTS
    scores = {
        m: sum(model_performance[m]) / len(model_performance[m]) if model_performance[m] else 0.25
        for m in model_performance
    }

    total = sum(scores.values())

    if total > 0:
        model_weights = {
            m: scores[m] / total for m in scores
        }

@app.get("/portfolio-backtest")
def portfolio_backtest(mode: str = "simple"):

    symbols = [
        "AAPL","TSLA","MSFT","GOOGL","AMZN",
        "RELIANCE.NS","TCS.NS","INFY.NS"
    ]

    initial_capital = 10000
    capital = initial_capital
    history = []
    # ✅ BENCHMARK TRACKING
    bh_capital = initial_capital   # Buy & Hold
    random_capital = initial_capital

    # =========================
    # SIMPLE BACKTEST (A)
    # =========================
    if mode == "simple":

        for s in symbols:
            try:
                df = get_stock_data(s)
                df = prepare_features(df)

                if len(df) < 60:
                    continue

                row = df.iloc[-2]

            # ✅ FEATURE PIPELINE (CORRECT)
                features_df = pd.DataFrame([row[["RSI", "MA20", "MA50", "momentum", "volatility"]]])
                features = scaler.transform(features_df)

            # ✅ MODEL PREDICTIONS
                dt_pred = dt_model.predict(features)[0]
                rf_pred = rf_model.predict(features)[0]
                lr_pred = lr_model.predict(features)[0]
                xgb_pred = xgb_model.predict(features)[0]

            # ✅ WEIGHTED SCORE
                score_ml = (
                dt_pred * model_weights["dt"] +
                rf_pred * model_weights["rf"] +
                lr_pred * model_weights["lr"] +
                xgb_pred * model_weights["xgb"]
                )

            # ✅ FINAL DECISION (NO HOLD)
                if score_ml >= 0.55:
                    prediction = 1
                elif score_ml < 0.45:
                    prediction = 0
                else:
                    continue

                current_price = float(df.iloc[-2]["Close"])
                next_price = float(df.iloc[-1]["Close"])
                # 🔵 BUY & HOLD
                bh_change = (next_price - current_price) / current_price
                bh_capital += bh_capital * bh_change

                # 🔴 RANDOM STRATEGY
                rand_pred = random.choice([0,1])
                rand_change = (
                    (next_price - current_price) / current_price
                    if rand_pred == 1
                    else (current_price - next_price) / current_price
                )
                random_capital += random_capital * rand_change

            # ✅ PNL CALCULATION
                slippage = 0.001  # 0.1%

                if prediction == 1:
                    entry = current_price * (1 + slippage)
                    exit = next_price * (1 - slippage)
                    change = (exit - entry) / entry
                else:
                    entry = current_price * (1 - slippage)
                    exit= next_price * (1 + slippage)
                    change = (entry- exit) / entry

                pnl = capital* 0.2 * change
                capital += pnl

                history.append(pnl)

            except:
                continue

    # =========================
    # ROLLING BACKTEST (B)
    # =========================

    # ✅ PRELOAD DATA (FIXES API ERROR)
    elif mode == "rolling":

        data_cache = {}

        for s in symbols:
            try:
                df = get_stock_data(s)
                df = prepare_features(df)
                data_cache[s] = df
            except:
                continue

        for step in range(50, 100):

            step_results = []
            bh_changes = []

            for s in symbols:
                try:
                    df = data_cache.get(s)

                    if df is None or len(df) < step + 2:
                        continue

                    row = df.iloc[step]

                    features_df = pd.DataFrame([row[["RSI", "MA20", "MA50", "momentum", "volatility"]]])
                    features = scaler.transform(features_df)

                    dt_pred = dt_model.predict(features)[0]
                    rf_pred = rf_model.predict(features)[0]
                    lr_pred = lr_model.predict(features)[0]
                    xgb_pred = xgb_model.predict(features)[0]

                    score_ml = (
                    dt_pred * model_weights["dt"] +
                    rf_pred * model_weights["rf"] +
                    lr_pred * model_weights["lr"] +
                    xgb_pred * model_weights["xgb"]
                    )

                    if score_ml >= 0.6:
                        prediction = 1
                    elif score_ml <= 0.4:
                        prediction = 0
                    else:
                        continue

                    price_now = float(df.iloc[step]["Close"])
                    price_next = float(df.iloc[step+1]["Close"])
                    
                    change = (
                    (price_next - price_now) / price_now
                    if prediction == 1
                    else (price_now - price_next) / price_now
                    )

                    step_results.append(change)

                    bh_changes.append((price_next - price_now) / price_now)
                    
                except:
                    continue

            if step_results:
                avg_change = sum(step_results) / len(step_results)
                capital += capital * avg_change
                history.append(capital * avg_change)

                # 🔵 BUY & HOLD (avg)
                # 🔵 TRUE BUY & HOLD (market movement only)
            if bh_changes:
                bh_avg = sum(bh_changes) / len(bh_changes)
                bh_capital += bh_capital * bh_avg

            # random
            rand_change = random.uniform(-0.02, 0.02)
            random_capital += random_capital * rand_change

           

    # =========================
    # METRICS
    # =========================
    total_return = ((capital - initial_capital) / initial_capital) * 100

    wins = len([h for h in history if h > 0])
    losses = len([h for h in history if h < 0])
    total = len(history)

    win_rate = (wins / total * 100) if total else 0

    # Max Drawdown
    peak = initial_capital
    drawdown = 0

    temp_cap = initial_capital
    for h in history:
        temp_cap += h
        peak = max(peak, temp_cap)
        dd = (temp_cap - peak) / peak
        drawdown = min(drawdown, dd)

    if history:
            returns = np.array(history)
            sharpe = round(np.mean(returns) / np.std(returns), 2) if np.std(returns) != 0 else 0
    else:
            sharpe = 0

    log_experiment(
    version="v1_baseline",
    change="ML + technical + sentiment (current system)",
    results={
        "return_pct": round((capital - 10000) / 10000 * 100, 2),
        "sharpe_ratio": sharpe,
        "win_rate": win_rate,
        "max_drawdown": drawdown
    }
    )

    return {
    "mode": mode,
    "your_strategy": {
        "final_capital": round(capital, 2),
        "return_pct": round(((capital - initial_capital) / initial_capital) * 100, 2)
    },
    "buy_and_hold": {
        "final_capital": round(bh_capital, 2),
        "return_pct": round(((bh_capital - initial_capital) / initial_capital) * 100, 2)
    },
    "random_strategy": {
        "final_capital": round(random_capital, 2),
        "return_pct": round(((random_capital - initial_capital) / initial_capital) * 100, 2)
    },
    "sharpe_ratio": sharpe,
    "win_rate": round(win_rate, 2),
    "max_drawdown": round(drawdown * 100, 2),
    "trades": total, 
    # ✅ LOG EXPERIMENT
    
}

# =========================
# PAPER TRADES
# =========================
@app.get("/paper-trades")
def get_paper_trades():
    return {
        "total_trades": len(paper_trades),
        "trades": paper_trades[-20:]
    }

# =========================
# PAPER PORTFOLIO
# =========================
@app.get("/paper-portfolio")
def paper_portfolio():

    total_positions = len(open_positions)

    return {
        "paper_balance": round(paper_balance, 2),
        "open_positions": total_positions,
        "positions": open_positions[-10:]
    }