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

print("STOCKSIGNAL v2 — DAVEY EDITION RUNNING")

dt_model = None; rf_model = None; lr_model = None; xgb_model = None
scaler = None; explainer = None

model_weights = {"dt": 0.25, "rf": 0.25, "lr": 0.25, "xgb": 0.25}
model_performance = {"dt": [], "rf": [], "lr": [], "xgb": []}
last_trade_time = None
paper_trades = []; paper_balance = 10000; open_positions = []
position_tracker = {}  # Davey timed exit tracker

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def log_experiment(version, change, results):
    path = "experiments.csv"
    write_header = not os.path.isfile(path)
    with open(path, mode="a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp","version","change","return_pct","sharpe","win_rate","drawdown"])
        w.writerow([datetime.now(), version, change,
                    results.get("return_pct"), results.get("sharpe_ratio"),
                    results.get("win_rate"), results.get("max_drawdown")])

def prepare_features(df):
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    # Basic
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / loss))
    df["volatility"] = df["Close"].pct_change().rolling(10).std()
    df["momentum"] = df["Close"].pct_change(10)
    df["momentum_short"] = df["Close"].pct_change(3)
    df["momentum_long"]  = df["Close"].pct_change(20)

    # ATR
    df["high_low"]   = df["High"] - df["Low"]
    df["high_close"] = abs(df["High"] - df["Close"].shift())
    df["low_close"]  = abs(df["Low"]  - df["Close"].shift())
    df["tr"]  = df[["high_low","high_close","low_close"]].max(axis=1)
    df["ATR"] = df["tr"].rolling(14).mean()

    # ADX — Davey Entry #4
    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    _plus    = plus_dm.where(plus_dm > minus_dm, 0)
    _minus   = minus_dm.where(minus_dm > plus_dm, 0)
    atr14    = df["tr"].rolling(14).mean()
    plus_di  = 100 * (_plus.rolling(14).mean()  / atr14.replace(0, np.nan))
    minus_di = 100 * (_minus.rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.rolling(14).mean()
    # ADX Rising — Davey Entry #8
    # ADX must be increasing for valid trend entries
    df["ADX_rising"] = df["ADX"] > df["ADX"].shift(3)


    # Bollinger Bands — Davey Entry #25
    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_pos"]   = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)

    # Liquidity filter
    df["vol_ma20"] = df["Volume"].rolling(20).mean() if "Volume" in df.columns else 1e9
    df["vol_ratio"] = df["Volume"] / df["vol_ma20"] if "Volume" in df.columns else 1.0

    # Volume breakout — Davey Entry #1
    # Strong breakouts need volume 1.5x above average
    df["vol_breakout"] = df["vol_ratio"] > 1.5

    # RSI Divergence — Davey Entry #12
    # Price new high but RSI lower high = bearish divergence (sell signal)
    # Price new low but RSI higher low = bullish divergence (buy signal)
    df["price_high_10"] = df["Close"].rolling(10).max()
    df["rsi_high_10"]   = df["RSI"].rolling(10).max()
    df["price_low_10"]  = df["Close"].rolling(10).min()
    df["rsi_low_10"]    = df["RSI"].rolling(10).min()
    df["bearish_divergence"] = (
        (df["Close"] >= df["price_high_10"] * 0.99) &   # price near 10-bar high
        (df["RSI"]   <= df["rsi_high_10"]   * 0.97)     # but RSI below its 10-bar high
    )
    df["bullish_divergence"] = (
        (df["Close"] <= df["price_low_10"]  * 1.01) &   # price near 10-bar low
        (df["RSI"]   >= df["rsi_low_10"]    * 1.03)     # but RSI above its 10-bar low
    )
    # Percentile of closes — Davey Exit #5
    df["close_pct_rank"] = df["Close"].rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    df.dropna(inplace=True)
    return df

@app.on_event("startup")
def train_model():
    global dt_model, rf_model, lr_model, xgb_model, scaler, explainer
    symbols = [
        "AAPL","TSLA","MSFT","GOOGL","AMZN",
        "JPM","GS","BAC","MS",
        "UBER","DIS","PYPL","CRM","ADBE",
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS"
    ]
    all_data = []
    for s in symbols:
        try:
            df = get_stock_data(s)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
            df = prepare_features(df)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"Skipped {s}: {e}")
            continue

    if not all_data:
        print("No training data"); return

    df = pd.concat(all_data, ignore_index=True)
    df["Close"] = df["Close"].astype(float)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(subset=["target"], inplace=True)

    X = df[["RSI","MA20","MA50","momentum","volatility"]]
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled[:-1]; y = y[:-1]

    dt_model  = DecisionTreeClassifier(max_depth=5, random_state=42)
    rf_model  = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    lr_model  = LogisticRegression(max_iter=1000, random_state=42)
    xgb_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric="logloss", random_state=42, verbosity=0)

    for m in [dt_model, rf_model, lr_model, xgb_model]:
        m.fit(X_scaled, y)

    explainer = shap.TreeExplainer(xgb_model)
    joblib.dump(dt_model,"dt.pkl"); joblib.dump(rf_model,"rf.pkl")
    joblib.dump(lr_model,"lr.pkl"); joblib.dump(xgb_model,"xgb.pkl")
    joblib.dump(scaler,"scaler.pkl")
    print("ALL MODELS TRAINED — v2 Davey Edition")

@app.get("/search")
def search_stock(query: str):
    data = [
        {"symbol":"AAPL","name":"Apple"},{"symbol":"TSLA","name":"Tesla"},
        {"symbol":"MSFT","name":"Microsoft"},{"symbol":"NVDA","name":"Nvidia"},
        {"symbol":"META","name":"Meta"},{"symbol":"AMZN","name":"Amazon"},
        {"symbol":"GOOGL","name":"Google"},{"symbol":"NFLX","name":"Netflix"},
        {"symbol":"AMD","name":"AMD"},{"symbol":"INTC","name":"Intel"},
        {"symbol":"ORCL","name":"Oracle"},{"symbol":"ADBE","name":"Adobe"},
        {"symbol":"CRM","name":"Salesforce"},{"symbol":"PYPL","name":"PayPal"},
        {"symbol":"UBER","name":"Uber"},{"symbol":"LYFT","name":"Lyft"},
        {"symbol":"BA","name":"Boeing"},{"symbol":"WMT","name":"Walmart"},
        {"symbol":"DIS","name":"Disney"},{"symbol":"NKE","name":"Nike"},
        {"symbol":"PFE","name":"Pfizer"},{"symbol":"KO","name":"Coca Cola"},
        {"symbol":"PEP","name":"Pepsi"},{"symbol":"MCD","name":"McDonalds"},
        {"symbol":"RELIANCE.NS","name":"Reliance"},{"symbol":"TCS.NS","name":"TCS"},
        {"symbol":"INFY.NS","name":"Infosys"},{"symbol":"HDFCBANK.NS","name":"HDFC Bank"},
        {"symbol":"ICICIBANK.NS","name":"ICICI Bank"},{"symbol":"SBIN.NS","name":"SBI"},
        {"symbol":"LT.NS","name":"L&T"},{"symbol":"ITC.NS","name":"ITC"},
        {"symbol":"WIPRO.NS","name":"Wipro"},{"symbol":"AXISBANK.NS","name":"Axis Bank"},
        {"symbol":"MARUTI.NS","name":"Maruti"},{"symbol":"TATAMOTORS.NS","name":"Tata Motors"},
        {"symbol":"ADANIENT.NS","name":"Adani Ent"},{"symbol":"HCLTECH.NS","name":"HCL Tech"},
        {"symbol":"SUNPHARMA.NS","name":"Sun Pharma"},{"symbol":"TATASTEEL.NS","name":"Tata Steel"},
        {"symbol":"BAJFINANCE.NS","name":"Bajaj Finance"},{"symbol":"ASIANPAINT.NS","name":"Asian Paints"},
        {"symbol":"ONGC.NS","name":"ONGC"},{"symbol":"NTPC.NS","name":"NTPC"},
        {"symbol":"POWERGRID.NS","name":"PowerGrid"},{"symbol":"ADANIPORTS.NS","name":"Adani Ports"},
        {"symbol":"ULTRACEMCO.NS","name":"UltraTech Cement"},
    ]
    return [s for s in data if query.lower() in s["symbol"].lower() or query.lower() in s["name"].lower()]

@app.get("/predict")
def predict(symbol: str = Query("AAPL")):
    global last_trade_time, paper_trades, open_positions, position_tracker

    # Sentiment
    try:
        res = requests.get(f"http://127.0.0.1:8001/sentiment?symbol={symbol}", timeout=2)
        sentiment_score = res.json().get("score", 0)
        sentiment_score_norm = round((sentiment_score + 1) * 50, 2)
    except:
        sentiment_score = 0; sentiment_score_norm = 50

    # Data
    df = get_stock_data(symbol)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df.dropna(inplace=True)
    if df.empty: return {"error": "Invalid symbol"}

    df = prepare_features(df)
    if df is None or df.empty or len(df) < 60:
        return {"error": "Not enough data"}

    latest = df.iloc[-1]
    prev   = df.iloc[-2]

    price      = float(latest["Close"])
    ma20       = float(latest["MA20"])
    ma50       = float(latest["MA50"])
    rsi        = float(latest["RSI"])
    momentum   = float(latest["momentum"])
    volatility = float(latest["volatility"])
    atr        = float(latest["ATR"])

    # Davey indicators
    adx           = float(latest.get("ADX", 25))
    bb_pos        = float(latest.get("BB_pos", 0.5))
    bb_upper      = float(latest.get("BB_upper", price * 1.02))
    bb_lower      = float(latest.get("BB_lower", price * 0.98))
    mom_short     = float(latest.get("momentum_short", 0))
    mom_long      = float(latest.get("momentum_long", 0))
    close_pct_rank= float(latest.get("close_pct_rank", 0.5))
    vol_ratio     = float(latest.get("vol_ratio", 1.0))
    vol_breakout      = bool(latest.get("vol_breakout", False))
    adx_rising        = bool(latest.get("ADX_rising", True))
    bearish_divergence = bool(latest.get("bearish_divergence", False))
    bullish_divergence = bool(latest.get("bullish_divergence", False))

    # ADX zones — Davey Entry #4
    adx_flat      = adx < 20
    adx_trending  = 20 <= adx <= 35
    adx_exhausted = adx > 40

    # Liquidity check — volume must be > 50% of 20-day average
    liquid = vol_ratio >= 0.5

    # ATR breakout filter — Davey Entry #5
    prev_close      = float(prev["Close"])
    atr_breakout_up = price > (prev_close + 1.0 * atr)
    atr_breakout_dn = price < (prev_close - 1.0 * atr)

    # Bollinger zones — Davey Entry #25
    bb_oversold   = bb_pos < 0.15
    bb_overbought = bb_pos > 0.85

    # Three Amigos — Davey Entry #27
    three_amigos_buy  = adx > 25 and rsi < 50 and mom_short > 0 and mom_long < 0
    three_amigos_sell = adx > 25 and rsi > 50 and mom_short < 0 and mom_long > 0

    # Quick Pullback — Davey Entry #32
    if len(df) >= 3:
        h0 = float(df["High"].iloc[-1]); h1 = float(df["High"].iloc[-2]); h2 = float(df["High"].iloc[-3])
        l0 = float(df["Low"].iloc[-1]);  l1 = float(df["Low"].iloc[-2]);  l2 = float(df["Low"].iloc[-3])
        pullback_buy  = (h2 > h1) and (price > h2)
        pullback_sell = (l2 < l1) and (price < l2)
    else:
        pullback_buy = pullback_sell = False

    # New High Consecutive — Davey Entry #37
    if len(df) >= 11:
        c0=float(df["Close"].iloc[-1]); c1=float(df["Close"].iloc[-2])
        c2=float(df["Close"].iloc[-3]); c3=float(df["Close"].iloc[-4])
        highest_10 = float(df["High"].iloc[-11:-1].max())
        lowest_10  = float(df["Low"].iloc[-11:-1].min())
        new_high_confirm = c0 > highest_10 and c0 > c1 and c0 > c3 and c1 > c2
        new_low_confirm  = c0 < lowest_10  and c0 < c1 and c0 < c3 and c1 < c2
    else:
        new_high_confirm = new_low_confirm = False

    # Consecutive Closes Exit — Davey Exit #6
    if len(df) >= 4:
        c0=float(df["Close"].iloc[-1]); c1=float(df["Close"].iloc[-2])
        c2=float(df["Close"].iloc[-3]); c3=float(df["Close"].iloc[-4])
        three_up_closes   = c0 > c1 > c2 > c3
        three_down_closes = c0 < c1 < c2 < c3
    else:
        three_up_closes = three_down_closes = False

    # Percentile Exit — Davey Exit #5
    pct_exit_long  = close_pct_rank < 0.5
    pct_exit_short = close_pct_rank > 0.5

    # Timed Exit — Davey Exit #3
    tracker = position_tracker.get(symbol, {})
    bars_held    = tracker.get("bars_held", 0)
    entry_price  = tracker.get("entry_price", price)
    peak_profit  = tracker.get("peak_profit", 0.0)
    direction    = tracker.get("direction", "")

    # Update peak profit — Davey Exit #8
    if direction == "BUY":
        current_profit = price - entry_price
    elif direction == "SELL":
        current_profit = entry_price - price
    else:
        current_profit = 0.0

    if current_profit > peak_profit:
        peak_profit = current_profit
        if symbol in position_tracker:
            position_tracker[symbol]["peak_profit"] = peak_profit

    # Don't Give It All Back — exit if gave back >50% of peak profit
    gave_back_too_much = (
        peak_profit > 0 and
        current_profit < peak_profit * 0.5 and
        bars_held > 3
    )

    timed_exit_triggered = bars_held >= 10

    # Original breakout levels
    recent_high = float(df["High"].rolling(20).max().iloc[-2])
    recent_low  = float(df["Low"].rolling(20).min().iloc[-2])
    is_breakout_up   = price > recent_high
    is_breakout_down = price < recent_low
    valid_breakout_up   = is_breakout_up   and price > float(prev["High"]) and rsi > 50
    valid_breakout_down = is_breakout_down and price < float(prev["Low"])  and rsi < 50

    # Regime
    if   ma20 > ma50 and volatility > 0.01: regime = "TREND_UP"
    elif ma20 < ma50 and volatility > 0.01: regime = "TREND_DOWN"
    else:                                   regime = "RANGE"

    # ML
    if dt_model is None: return {"error": "Models not trained yet"}
    feat_df  = pd.DataFrame([latest[["RSI","MA20","MA50","momentum","volatility"]]])
    features = scaler.transform(feat_df)
    dt_pred  = dt_model.predict(features)[0]
    rf_pred  = rf_model.predict(features)[0]
    lr_pred  = lr_model.predict(features)[0]
    xgb_pred = xgb_model.predict(features)[0]

    dt_prob  = dt_model.predict_proba(features)[0][1]
    rf_prob  = rf_model.predict_proba(features)[0][1]
    lr_prob  = lr_model.predict_proba(features)[0][1]
    xgb_prob = xgb_model.predict_proba(features)[0][1]

    score_ml = (dt_pred*model_weights["dt"] + rf_pred*model_weights["rf"] +
                lr_pred*model_weights["lr"] + xgb_pred*model_weights["xgb"])
    ml_confidence = round((dt_prob*model_weights["dt"] + rf_prob*model_weights["rf"] +
                           lr_prob*model_weights["lr"] + xgb_prob*model_weights["xgb"]) * 100, 2)

    shap_values = explainer.shap_values(features)[0]
    feat_names  = ["RSI","MA20","MA50","momentum","volatility"]
    shap_sorted = dict(sorted({feat_names[i]: float(shap_values[i]) for i in range(5)}.items(),
                               key=lambda x: abs(x[1]), reverse=True))
    votes     = dt_pred + rf_pred + lr_pred + xgb_pred
    agreement = f"{votes}/4 models bullish"

    # Signals
    signals, score = [], 0
    if ma20 > ma50: signals.append("MA20 > MA50 (Bullish)"); score += 1
    else:           signals.append("MA20 < MA50 (Bearish)"); score -= 1
    if rsi < 30:    signals.append("RSI Oversold"); score += 2
    elif rsi > 70:  signals.append("RSI Overbought"); score -= 2
    else:           signals.append("RSI Neutral")
    if momentum > 0: signals.append("Momentum Positive"); score += 1
    else:            signals.append("Momentum Negative"); score -= 1

    # Davey signals
    if adx_flat:      signals.append(f"ADX Low {adx:.1f} — Breakout Setup [Davey #4]"); score += 1
    elif adx_trending: signals.append(f"ADX Trending {adx:.1f} — Momentum Zone [Davey #4]")
    elif adx_exhausted: signals.append(f"ADX Exhausted {adx:.1f} — Avoid [Davey #4]"); score -= 2
    if bb_oversold:   signals.append("Near Bollinger Lower Band [Davey #25]"); score += 1
    elif bb_overbought: signals.append("Near Bollinger Upper Band [Davey #25]"); score -= 1
    if three_amigos_buy:  signals.append("Three Amigos BUY [Davey #27]"); score += 2
    elif three_amigos_sell: signals.append("Three Amigos SELL [Davey #27]"); score -= 2
    if pullback_buy:  signals.append("Quick Pullback Continuation [Davey #32]"); score += 2
    elif pullback_sell: signals.append("Quick Pullback SELL [Davey #32]"); score -= 2
    if new_high_confirm: signals.append("New High 4/4 Momentum Confirmed [Davey #37]"); score += 2
    elif new_low_confirm: signals.append("New Low 4/4 Momentum Confirmed [Davey #37]"); score -= 2
    if atr_breakout_up: signals.append("ATR Significant Breakout UP [Davey #5]"); score += 1
    elif atr_breakout_dn: signals.append("ATR Significant Breakout DOWN [Davey #5]"); score -= 1
    if not liquid:    signals.append("LOW LIQUIDITY — Volume below average"); score -= 1
    if three_up_closes:   signals.append("⚠️ 3 Consecutive Up Closes — Exit Signal [Davey #6]")
    if three_down_closes: signals.append("⚠️ 3 Consecutive Down Closes — Exit Signal [Davey #6]")
    if timed_exit_triggered: signals.append(f"⚠️ Held {bars_held} days — Timed Exit [Davey #3]")
    if gave_back_too_much:
        signals.append(f"⚠️ Gave back >50% of peak profit — Exit [Davey #8]")
    if vol_breakout:
        signals.append("Volume Breakout — 1.5x average volume [Davey #1]"); score += 1
    if adx_rising:
        signals.append("ADX Rising — Trend strengthening [Davey #8]"); score += 1
    if bullish_divergence:
        signals.append("Bullish RSI Divergence — Price low, RSI higher [Davey #12]"); score += 2
    if bearish_divergence:
        signals.append("Bearish RSI Divergence — Price high, RSI lower [Davey #12]"); score -= 2

    technical_score = round(((score + 8) / 16) * 100, 2)
    confidence = round(0.5 * ml_confidence + 0.3 * technical_score + 0.2 * sentiment_score_norm, 2)

    # ═══════════════════════════════════════
    # DECISION ENGINE — Full Davey Stack
    # ═══════════════════════════════════════
    prediction = "HOLD"

    # L1: Timed exit override
    if timed_exit_triggered:
        prediction = "HOLD"
        position_tracker.pop(symbol, None)

    # L2: ADX exhausted — no new entries
    elif adx_exhausted:
        prediction = "HOLD"

    # L3: Liquidity — skip illiquid stocks
    elif not liquid:
        prediction = "HOLD"

    else:
        if regime == "TREND_UP":
            buy = False
            # A: Breakout + ADX + ATR + Volume (Davey #1 + #4 + #5)
            if valid_breakout_up and (adx_flat or adx_trending) and atr_breakout_up and vol_breakout: buy = True
            # B: Three Amigos + ADX Rising (Davey #27 + #8)
            if three_amigos_buy and score_ml >= 0.5 and adx_rising: buy = True
            # C: Quick Pullback (Davey #32)
            if pullback_buy and score_ml >= 0.5 and (adx_flat or adx_trending): buy = True
            # D: New High confirmation (Davey #37)
            if new_high_confirm and score_ml >= 0.6 and vol_breakout: buy = True
            # E: Bullish divergence — strong reversal signal (Davey #12)
            if bullish_divergence and score_ml >= 0.5: buy = True
            if buy: prediction = "BUY"

        elif regime == "TREND_DOWN":
            sell = False
            if valid_breakout_down and (adx_flat or adx_trending) and atr_breakout_dn and vol_breakout: sell = True
            if three_amigos_sell and score_ml <= 0.5 and adx_rising: sell = True
            if pullback_sell and score_ml <= 0.5 and (adx_flat or adx_trending): sell = True
            if new_low_confirm and score_ml <= 0.4 and vol_breakout: sell = True
            # Bearish divergence — strong reversal signal (Davey #12)
            if bearish_divergence and score_ml <= 0.5: sell = True
            if sell: prediction = "SELL"

        elif regime == "RANGE":
            # Bollinger + RSI double confirmation (Davey #25)
            if rsi < 30 and bb_oversold:    prediction = "BUY"
            elif rsi > 70 and bb_overbought: prediction = "SELL"
            elif rsi < 30:                  prediction = "BUY"
            elif rsi > 70:                  prediction = "SELL"

        # Exit overrides
        if prediction == "BUY"  and three_down_closes: prediction = "HOLD"
        if prediction == "SELL" and three_up_closes:   prediction = "HOLD"
        if prediction == "BUY"  and pct_exit_long and bars_held > 5: prediction = "HOLD"

    # Don't Give It All Back exit (Davey Exit #8)
    if gave_back_too_much: prediction = "HOLD"
    # Bearish divergence overrides BUY (Davey #12)
    if prediction == "BUY"  and bearish_divergence: prediction = "HOLD"
    if prediction == "SELL" and bullish_divergence: prediction = "HOLD"

    # Standard filters
    if confidence < 52:    prediction = "HOLD"
    if volatility < 0.005: prediction = "HOLD"
    if prediction == "BUY"  and sentiment_score < -0.2: prediction = "HOLD"
    if prediction == "SELL" and sentiment_score >  0.2: prediction = "HOLD"
    if last_trade_time and datetime.now() - last_trade_time < timedelta(minutes=30):
        prediction = "HOLD"

    # Update position tracker
    if prediction in ("BUY","SELL"):
        last_trade_time = datetime.now()
        position_tracker[symbol] = {"bars_held": 0, "entry_price": price, "direction": prediction}
    elif symbol in position_tracker:
        position_tracker[symbol]["bars_held"] += 1

    # Trade plan
    support    = float(df["Low"].rolling(20).min().iloc[-1])
    resistance = float(df["High"].rolling(20).max().iloc[-1])
    capital    = 10000; risk_per_trade = 0.02

    if prediction == "BUY":
        entry = price; stop_loss = round(price - 1.5*atr, 2); target = round(price + 2.0*atr, 2)
    elif prediction == "SELL":
        entry = price; stop_loss = round(price + 1.5*atr, 2); target = round(price - 2.0*atr, 2)
    else:
        entry = stop_loss = target = "-"

    if prediction != "HOLD":
        risk_pu = abs(entry - stop_loss)
        position_size = int((capital * risk_per_trade) / risk_pu) if risk_pu > 0 else 0
        capital_used  = round(position_size * entry, 2)
        rr = round(abs(target - entry) / abs(entry - stop_loss), 2) if abs(entry - stop_loss) else 0
    else:
        position_size = capital_used = 0; rr = "-"

    # Paper trade
    if prediction in ("BUY","SELL"):
        trade = {"time": str(datetime.now()), "symbol": symbol, "prediction": prediction,
                 "price": round(price,2), "confidence": confidence, "regime": regime,
                 "target": target, "stop_loss": stop_loss, "position_size": position_size}
        paper_trades.append(trade); open_positions.append(trade)

    # AI
    trend = "Uptrend" if ma20 > ma50 else "Downtrend"
    reason = []
    if ma20 > ma50:          reason.append("Bullish trend (MA crossover)")
    else:                    reason.append("Bearish trend (MA crossover)")
    if rsi < 30:             reason.append("Stock oversold")
    elif rsi > 70:           reason.append("Stock overbought")
    if momentum > 0:         reason.append("Positive momentum")
    else:                    reason.append("Negative momentum")
    if three_amigos_buy:     reason.append("Three Amigos BUY pattern [Davey]")
    if pullback_buy:         reason.append("Quick Pullback continuation [Davey]")
    if new_high_confirm:     reason.append("New high 4/4 momentum [Davey]")
    if adx_flat:             reason.append("ADX low — clean breakout setup [Davey]")
    if not liquid:           reason.append("LOW LIQUIDITY — trade with caution")

    ai_analysis = analyze_stock_ai(symbol, trend, round(rsi,2), prediction, confidence)
    ai_analysis += "\n\nReasons:\n" + ", ".join(reason)

    info = get_stock_info(symbol)
    best_model = max([("Decision Tree",dt_prob),("Random Forest",rf_prob),
                      ("Logistic",lr_prob),("XGBoost",xgb_prob)], key=lambda x: x[1])[0]

    return {
        "symbol": symbol, "sentiment_score": sentiment_score,
        "best_model": best_model,
        "company": info.get("name", symbol),
        "price": round(price, 2), "prediction": prediction, "confidence": confidence,
        "confidence_breakdown": {"ml_confidence": ml_confidence,
                                 "technical_score": technical_score, "agreement": agreement},
        "signals": signals,
        "trade_plan": {"entry": entry, "stop_loss": stop_loss, "target": target,
                       "risk_reward": rr, "position_size": position_size,
                       "capital_used": capital_used, "risk_per_trade_pct": 2},
        "ai_analysis": ai_analysis,
        "sentiment": {"score_raw": sentiment_score, "score_normalized": sentiment_score_norm},
        "breakout": {"recent_high": recent_high, "recent_low": recent_low,
                     "is_breakout_up": is_breakout_up, "is_breakout_down": is_breakout_down},
        "breakout_quality": {"valid_up": valid_breakout_up, "valid_down": valid_breakout_down},
        "davey_signals": {
            "adx": round(adx,2),
            "adx_zone": "FLAT" if adx_flat else "TRENDING" if adx_trending else "EXHAUSTED",
            "three_amigos_buy": three_amigos_buy, "three_amigos_sell": three_amigos_sell,
            "pullback_buy": pullback_buy, "pullback_sell": pullback_sell,
            "new_high_confirm": new_high_confirm, "new_low_confirm": new_low_confirm,
            "atr_breakout_up": atr_breakout_up, "atr_breakout_dn": atr_breakout_dn,
            "bb_pos": round(bb_pos,3),
            "bb_zone": "OVERSOLD" if bb_oversold else "OVERBOUGHT" if bb_overbought else "NORMAL",
            "three_up_closes": three_up_closes, "three_down_closes": three_down_closes,
            "timed_exit": timed_exit_triggered, "bars_held": bars_held,
            "liquid": liquid, "vol_ratio": round(vol_ratio,2),
        },
        "models": {"decision_tree": "BUY" if dt_pred==1 else "SELL",
                   "random_forest": "BUY" if rf_pred==1 else "SELL",
                   "logistic":      "BUY" if lr_pred==1 else "SELL",
                   "xgboost":       "BUY" if xgb_pred==1 else "SELL"},
        "votes": int(votes), "regime": regime, "score": int(score),
        "explainability": shap_sorted,
        "market_structure": {"support": round(support,2), "resistance": round(resistance,2), "atr": round(atr,2)},
    }

@app.get("/top-opportunity")
def top_opportunity():
    symbols = [
        "AAPL","TSLA","MSFT","GOOGL","AMZN","META","NVDA","AMD","NFLX","INTC",
        "BABA","ORCL","UBER","DIS","PYPL","CRM","ADBE","CSCO","QCOM","SHOP",
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "SBIN.NS","ITC.NS","LT.NS","WIPRO.NS","AXISBANK.NS",
        "BAJFINANCE.NS","MARUTI.NS","ADANIENT.NS","ONGC.NS"
    ]
    results = []
    for s in symbols:
        try:
            res = predict(s)
            if "confidence" in res and isinstance(res["confidence"], (int, float)):
                results.append({"symbol": res["symbol"], "confidence": res["confidence"],
                                 "prediction": res["prediction"]})
        except: continue
    results.sort(key=lambda x: x["confidence"], reverse=True)
    top5 = results[:5]
    total_conf = sum(s["confidence"] for s in top5) or 1
    capital = 10000
    return {
        "total_capital": capital,
        "portfolio": [{"symbol": s["symbol"], "prediction": s["prediction"],
                        "confidence": s["confidence"],
                        "weight": round(s["confidence"]/total_conf, 3),
                        "capital_allocated": round(capital * s["confidence"]/total_conf, 2)}
                       for s in top5]
    }

@app.get("/backtest")
def backtest(symbol: str = "AAPL"):
    df = get_stock_data(symbol)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty: return {"error": "Invalid symbol"}
    df = prepare_features(df)
    if len(df) < 100: return {"error": "Not enough data"}
    wins = losses = profit = 0
    for i in range(50, len(df)-1):
        row = df.iloc[i].copy()
        feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
        f = scaler.transform(feat)
        score_ml = (dt_model.predict(f)[0]*model_weights["dt"] +
                    rf_model.predict(f)[0]*model_weights["rf"] +
                    lr_model.predict(f)[0]*model_weights["lr"] +
                    xgb_model.predict(f)[0]*model_weights["xgb"])
        pred = 1 if score_ml >= 0.5 else 0
        cur = float(row["Close"]); nxt = float(df.iloc[i+1]["Close"])
        actual = 1 if nxt > cur else 0
        if pred == actual: wins += 1
        else: losses += 1
        change = (nxt-cur)/cur if pred==1 else (cur-nxt)/cur
        profit += 1000 * change
    total = wins + losses
    return {"symbol": symbol, "accuracy": round(wins/total*100,2) if total else 0,
            "total_trades": total, "wins": wins, "losses": losses,
            "profit": round(profit,2), "final_capital": round(10000+profit,2)}

@app.get("/simulate-profit")
def simulate_profit(symbol: str = "AAPL"):
    df = get_stock_data(symbol)
    df = prepare_features(df)
    capital = 10000
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
        f = scaler.transform(feat)
        pred = rf_model.predict(f)[0]
        nxt = float(df.iloc[i+1]["Close"]); cur = float(row["Close"])
        capital += (nxt-cur) if pred==1 else (cur-nxt)
    return {"final_capital": round(capital,2), "profit": round(capital-10000,2)}

@app.get("/model-stats")
def model_stats(symbol: str = "AAPL"):
    from sklearn.metrics import confusion_matrix, accuracy_score
    df = get_stock_data(symbol)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    if df.empty: return {"error": "Invalid symbol"}
    df = prepare_features(df)
    if len(df) < 100: return {"error": "Not enough data"}
    y_true=[]; dt_p=[]; rf_p=[]; lr_p=[]; xgb_p=[]
    for i in range(50, len(df)-1):
        row = df.iloc[i].copy()
        feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
        f = scaler.transform(feat)
        actual = 1 if float(df.iloc[i+1]["Close"]) > float(row["Close"]) else 0
        y_true.append(actual)
        dt_p.append(dt_model.predict(f)[0]); rf_p.append(rf_model.predict(f)[0])
        lr_p.append(lr_model.predict(f)[0]); xgb_p.append(xgb_model.predict(f)[0])
    global model_weights
    accs = {"dt": accuracy_score(y_true,dt_p), "rf": accuracy_score(y_true,rf_p),
            "lr": accuracy_score(y_true,lr_p), "xgb": accuracy_score(y_true,xgb_p)}
    total = sum(accs.values())
    model_weights = {k: v/total for k,v in accs.items()}
    return {"decision_tree_accuracy": round(accs["dt"]*100,2),
            "random_forest_accuracy": round(accs["rf"]*100,2),
            "logistic_accuracy":      round(accs["lr"]*100,2),
            "xgboost_accuracy":       round(accs["xgb"]*100,2),
            "weights": model_weights,
            "decision_tree_cm": confusion_matrix(y_true,dt_p).tolist(),
            "random_forest_cm": confusion_matrix(y_true,rf_p).tolist(),
            "logistic_cm":      confusion_matrix(y_true,lr_p).tolist()}

@app.get("/portfolio-backtest")
def portfolio_backtest(mode: str = "simple"):
    symbols = ["AAPL","TSLA","MSFT","GOOGL","AMZN","RELIANCE.NS","TCS.NS","INFY.NS"]
    initial = capital = bh = rand = 10000.0; history = []
    if mode == "simple":
        for s in symbols:
            try:
                df = get_stock_data(s); df = prepare_features(df)
                if len(df) < 60: continue
                row = df.iloc[-2]
                feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
                f = scaler.transform(feat)
                sc = (dt_model.predict(f)[0]*model_weights["dt"] +
                      rf_model.predict(f)[0]*model_weights["rf"] +
                      lr_model.predict(f)[0]*model_weights["lr"] +
                      xgb_model.predict(f)[0]*model_weights["xgb"])
                if sc >= 0.55: pred=1
                elif sc < 0.45: pred=0
                else: continue
                cur=float(df.iloc[-2]["Close"]); nxt=float(df.iloc[-1]["Close"])
                slip=0.001
                change = ((nxt*(1-slip)-cur*(1+slip))/(cur*(1+slip))) if pred==1 else ((cur*(1-slip)-nxt*(1+slip))/(cur*(1-slip)))
                pnl = capital*0.2*change; capital += pnl; history.append(pnl)
                bh   += bh   * ((nxt-cur)/cur)
                rand += rand * random.uniform(-0.02, 0.02)
            except: continue
    elif mode == "rolling":
        cache = {}
        for s in symbols:
            try:
                df = get_stock_data(s); df = prepare_features(df); cache[s] = df
            except: continue
        for step in range(50,100):
            changes=[]; bh_ch=[]
            for s in symbols:
                df = cache.get(s)
                if df is None or len(df) < step+2: continue
                try:
                    row = df.iloc[step]
                    feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
                    f = scaler.transform(feat)
                    sc = (dt_model.predict(f)[0]*model_weights["dt"] +
                          rf_model.predict(f)[0]*model_weights["rf"] +
                          lr_model.predict(f)[0]*model_weights["lr"] +
                          xgb_model.predict(f)[0]*model_weights["xgb"])
                    if sc >= 0.6: pred=1
                    elif sc <= 0.4: pred=0
                    else: continue
                    p0=float(df.iloc[step]["Close"]); p1=float(df.iloc[step+1]["Close"])
                    changes.append((p1-p0)/p0 if pred==1 else (p0-p1)/p0)
                    bh_ch.append((p1-p0)/p0)
                except: continue
            if changes: avg=sum(changes)/len(changes); capital+=capital*avg; history.append(capital*avg)
            if bh_ch: bh += bh*(sum(bh_ch)/len(bh_ch))
            rand += rand*random.uniform(-0.02,0.02)
    wins=sum(1 for h in history if h>0); total=len(history)
    win_rate=(wins/total*100) if total else 0
    peak=cap_t=float(initial); dd=0.0
    for h in history:
        cap_t+=h; peak=max(peak,cap_t); dd=min(dd,(cap_t-peak)/peak)
    arr=np.array(history)
    sharpe=round(np.mean(arr)/np.std(arr),2) if len(arr)>1 and np.std(arr)!=0 else 0
    log_experiment("v2_davey",f"mode={mode}",{
        "return_pct": round((capital-initial)/initial*100,2),
        "sharpe_ratio": sharpe, "win_rate": win_rate, "max_drawdown": dd})
    return {"mode": mode,
            "your_strategy": {"final_capital": round(capital,2), "return_pct": round((capital-initial)/initial*100,2)},
            "buy_and_hold":   {"final_capital": round(bh,2),     "return_pct": round((bh-initial)/initial*100,2)},
            "random_strategy":{"final_capital": round(rand,2),   "return_pct": round((rand-initial)/initial*100,2)},
            "sharpe_ratio": sharpe, "win_rate": round(win_rate,2),
            "max_drawdown": round(dd*100,2), "trades": total}

@app.get("/paper-trades")
def get_paper_trades():
    return {"total_trades": len(paper_trades), "trades": paper_trades[-20:]}

@app.get("/paper-portfolio")
def paper_portfolio():
    return {"paper_balance": round(paper_balance,2),
            "open_positions": len(open_positions), "positions": open_positions[-10:]}

def update_model_performance(symbol, last_preds, next_price):
    global model_performance, model_weights
    actual = 1 if next_price > last_preds["price"] else 0
    for model in ["dt", "rf", "lr", "xgb"]:
        pred = last_preds[model]
        model_performance[model].append(1 if pred == actual else 0)
        model_performance[model] = model_performance[model][-50:]
    scores = {m: sum(model_performance[m])/len(model_performance[m]) 
              if model_performance[m] else 0.25 for m in model_performance}
    total = sum(scores.values())
    if total > 0:
        model_weights = {m: scores[m]/total for m in scores}
