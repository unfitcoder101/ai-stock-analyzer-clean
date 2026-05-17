import joblib
import numpy as np
import random
import pandas as pd
import requests
import csv
import os
import yfinance as yf
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

print("STOCKSIGNAL v3 — DAVEY + O'NEIL + ALDRIDGE EDITION")

dt_model = None; rf_model = None; lr_model = None; xgb_model = None
scaler = None; explainer = None

model_weights = {"dt": 0.25, "rf": 0.25, "lr": 0.25, "xgb": 0.25}
model_performance = {"dt": [], "rf": [], "lr": [], "xgb": []}
last_trade_time = None
paper_trades = []; paper_balance = 10000; open_positions = []
position_tracker = {}

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

    # ── Basic indicators ───────────────────────────────────────────────
    df["MA20"]  = df["Close"].rolling(20).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    df["MA150"] = df["Close"].rolling(150).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    # MA200 trend — is it rising over last 21 days
    df["MA200_rising"] = df["MA200"] > df["MA200"].shift(21)
    # 52-week low for Minervini condition 6
    df["low_52w"]           = df["Low"].rolling(252).min()
    df["pct_from_52w_low"]  = (df["Close"] - df["low_52w"]) / df["low_52w"] * 100
    df["above_30pct_52w_low"] = df["pct_from_52w_low"] >= 30
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]        = 100 - (100 / (1 + gain / loss))
    df["volatility"] = df["Close"].pct_change().rolling(10).std()
    df["momentum"]       = df["Close"].pct_change(10)
    df["momentum_short"] = df["Close"].pct_change(3)
    df["momentum_long"]  = df["Close"].pct_change(20)

    # ── ATR ────────────────────────────────────────────────────────────
    df["high_low"]   = df["High"] - df["Low"]
    df["high_close"] = abs(df["High"] - df["Close"].shift())
    df["low_close"]  = abs(df["Low"]  - df["Close"].shift())
    df["tr"]         = df[["high_low","high_close","low_close"]].max(axis=1)
    df["ATR"]        = df["tr"].rolling(14).mean()

    # ── ADX — Davey Entry #4 ───────────────────────────────────────────
    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    _plus    = plus_dm.where(plus_dm > minus_dm, 0)
    _minus   = minus_dm.where(minus_dm > plus_dm, 0)
    atr14    = df["tr"].rolling(14).mean()
    plus_di  = 100 * (_plus.rolling(14).mean()  / atr14.replace(0, np.nan))
    minus_di = 100 * (_minus.rolling(14).mean() / atr14.replace(0, np.nan))
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"]        = dx.rolling(14).mean()
    df["ADX_rising"] = df["ADX"] > df["ADX"].shift(3)

    # ── Bollinger Bands — Davey Entry #25 ─────────────────────────────
    bb_mid         = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std
    df["BB_pos"]   = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)

    # ── Volume & Liquidity ─────────────────────────────────────────────
    df["vol_ma20"]         = df["Volume"].rolling(20).mean() if "Volume" in df.columns else 1e9
    df["vol_ratio"]        = df["Volume"] / df["vol_ma20"]   if "Volume" in df.columns else 1.0
    df["vol_breakout"]     = df["vol_ratio"] > 1.5   # Davey Entry #1
    df["vol_breakout_oneil"] = df["vol_ratio"] > 1.4 # O'Neil: 40% above

    # ── Money Flow Index — Davey Entry #23 ────────────────────────────
    if "Volume" in df.columns:
        tp            = (df["High"] + df["Low"] + df["Close"]) / 3
        raw_mf        = tp * df["Volume"]
        pos_flow      = raw_mf.where(tp > tp.shift(1), 0)
        neg_flow      = raw_mf.where(tp < tp.shift(1), 0)
        mfi_ratio     = pos_flow.rolling(14).sum() / neg_flow.rolling(14).sum().replace(0, np.nan)
        df["MFI"]     = 100 - (100 / (1 + mfi_ratio))
    else:
        df["MFI"] = 50.0

    # ── Stochastic Oscillator — Davey Entry #24 ───────────────────────
    lowest_14      = df["Low"].rolling(14).min()
    highest_14     = df["High"].rolling(14).max()
    df["stoch_k"]  = 100 * (df["Close"] - lowest_14) / (highest_14 - lowest_14).replace(0, np.nan)
    df["stoch_d"]  = df["stoch_k"].rolling(3).mean()

    # ── Range Contraction — Davey Entry #41 ───────────────────────────
    df["daily_range"]       = df["High"] - df["Low"]
    df["range_contracting"] = df["daily_range"] < df["daily_range"].shift(1)

    # ── Big Tail Bars — Davey Entry #36 ───────────────────────────────
    bar_range  = df["High"] - df["Low"]
    lower_tail = df["Close"] - df["Low"]
    upper_tail = df["High"] - df["Close"]
    df["bull_tail_bar"] = (
        (lower_tail > bar_range * 0.6) &
        (df["High"] >= df["High"].shift(1)) &
        (df["Close"] > df["Open"])
    )
    df["bear_tail_bar"] = (
        (upper_tail > bar_range * 0.6) &
        (df["Low"]  <= df["Low"].shift(1)) &
        (df["Close"] < df["Open"])
    )

    # ── RSI Divergence — Davey Entry #12 ──────────────────────────────
    df["price_high_10"]      = df["Close"].rolling(10).max()
    df["rsi_high_10"]        = df["RSI"].rolling(10).max()
    df["price_low_10"]       = df["Close"].rolling(10).min()
    df["rsi_low_10"]         = df["RSI"].rolling(10).min()
    df["bearish_divergence"] = (
        (df["Close"] >= df["price_high_10"] * 0.99) &
        (df["RSI"]   <= df["rsi_high_10"]   * 0.97)
    )
    df["bullish_divergence"] = (
        (df["Close"] <= df["price_low_10"]  * 1.01) &
        (df["RSI"]   >= df["rsi_low_10"]    * 1.03)
    )

    # ── Percentile Exit — Davey Exit #5 ───────────────────────────────
    df["close_pct_rank"] = df["Close"].rolling(5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # ── Z-Score — Aldridge Chapter 13 ─────────────────────────────────
    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std  = df["Close"].rolling(20).std()
    df["zscore"] = (df["Close"] - rolling_mean) / rolling_std.replace(0, np.nan)

    # ── O'Neil CANSLIM — L: Leader ────────────────────────────────────
    df["is_leader"] = (df["Close"] > df["MA50"]) & (df["momentum"] > 0)

    # ── Minervini Trend Template — all 8 conditions ───────────────────
    df["tt1"] = df["Close"] > df["MA150"]                          # above 150MA
    df["tt2"] = df["Close"] > df["MA200"]                          # above 200MA
    df["tt3"] = df["MA150"] > df["MA200"]                          # 150 above 200
    df["tt4"] = df["MA200_rising"]                                  # 200MA trending up
    df["tt5"] = df["MA50"]  > df["MA150"]                          # 50 above 150
    df["tt6"] = df["MA50"]  > df["MA200"]                          # 50 above 200
    df["tt7"] = df["Close"] > df["MA50"]                           # price above 50MA
    df["tt8"] = df["above_30pct_52w_low"]                          # 30%+ above 52w low
    df["tt9"] = df["pct_from_52w_high"] > -25                      # within 25% of 52w high
    # All 8 conditions = confirmed Stage 2 uptrend
    df["trend_template"] = (
        df["tt1"] & df["tt2"] & df["tt3"] & df["tt4"] &
        df["tt5"] & df["tt6"] & df["tt7"] & df["tt8"] & df["tt9"]
    )
    # Count how many conditions pass (0-9)
    df["trend_score"] = (
        df["tt1"].astype(int) + df["tt2"].astype(int) +
        df["tt3"].astype(int) + df["tt4"].astype(int) +
        df["tt5"].astype(int) + df["tt6"].astype(int) +
        df["tt7"].astype(int) + df["tt8"].astype(int) +
        df["tt9"].astype(int)
    )

    # ── Weinstein Stage Analysis ──────────────────────────────────────
    # Stage 2 = price above rising 30-week (150-day) MA
    df["weinstein_stage2"] = (
        (df["Close"] > df["MA150"]) &
        (df["MA150"] > df["MA150"].shift(10))  # MA150 rising
    )
    # Stage 4 = price below falling MA150 — never buy
    df["weinstein_stage4"] = (
        (df["Close"] < df["MA150"]) &
        (df["MA150"] < df["MA150"].shift(10))
    )

    # ── VCP — Volatility Contraction Pattern (Minervini) ─────────────
    # Range contracting = current 5-day range < 50% of 20-day range
    df["range_5d"]  = df["High"].rolling(5).max()  - df["Low"].rolling(5).min()
    df["range_20d"] = df["High"].rolling(20).max() - df["Low"].rolling(20).min()
    df["vcp_contracting"] = df["range_5d"] < (df["range_20d"] * 0.5)
    # Volume drying up at pivot = volume below 50-day average
    df["vol_ma50"] = df["Volume"].rolling(50).mean() if "Volume" in df.columns else 1e9
    df["vol_dry"]  = df["Volume"] < df["vol_ma50"]   if "Volume" in df.columns else False
    # VCP confirmed = range contracting AND volume drying up
    df["vcp_setup"] = df["vcp_contracting"] & df["vol_dry"]

    # ── O'Neil CANSLIM — I: Institutional Sponsorship proxy ───────────
    if "Volume" in df.columns:
        up_on_vol          = (df["Close"] > df["Close"].shift(1)) & (df["vol_ratio"] > 1.0)
        dn_on_vol          = (df["Close"] < df["Close"].shift(1)) & (df["vol_ratio"] > 1.0)
        df["accum_days"]   = up_on_vol.rolling(20).sum()
        df["distrib_days"] = dn_on_vol.rolling(20).sum()
        df["institutional_buying"] = df["accum_days"] > df["distrib_days"]
        # Distribution day counter — O'Neil M factor
        df["distrib_day"]       = (df["Close"] < df["Close"].shift(1)) & (df["Volume"] > df["Volume"].shift(1))
        df["distrib_count_25d"] = df["distrib_day"].rolling(25).sum()
    else:
        df["accum_days"]         = 10
        df["distrib_days"]       = 5
        df["institutional_buying"] = True
        df["distrib_day"]        = False
        df["distrib_count_25d"]  = 0

    # ── O'Neil CANSLIM — M: Follow-Through Day ────────────────────────
    df["rally_day"]      = df["Close"] > df["Close"].shift(1)
    df["rally_streak"]   = df["rally_day"].rolling(4).sum()
    df["follow_through"] = (df["rally_streak"] >= 4) & (df["vol_ratio"] > 1.0)

    # ── O'Neil 52-Week High Filter ─────────────────────────────────────
    df["high_52w"]          = df["High"].rolling(252).max()
    df["pct_from_52w_high"] = (df["Close"] - df["high_52w"]) / df["high_52w"] * 100
    df["near_52w_high"]     = df["pct_from_52w_high"] > -25

    df.dropna(inplace=True)
    return df


@app.on_event("startup")
def load_models():
    global dt_model, rf_model, lr_model, xgb_model, scaler, explainer
    try:
        dt_model  = joblib.load("dt.pkl")
        rf_model  = joblib.load("rf.pkl")
        lr_model  = joblib.load("lr.pkl")
        xgb_model = joblib.load("xgb.pkl")
        scaler    = joblib.load("scaler.pkl")
        print("✅ Pre-trained models loaded")
    except Exception as e:
        print(f"No saved models found, training now: {e}")
        _train_from_scratch()

def _train_from_scratch():
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
    if not all_data:
        print("No training data"); return
    df = pd.concat(all_data, ignore_index=True)
    df["Close"] = df["Close"].astype(float)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(subset=["target"], inplace=True)
    X = df[["RSI","MA20","MA50","momentum","volatility"]]
    y = df["target"]
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X)
    X_scaled  = X_scaled[:-1]; y = y[:-1]
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
    print("✅ Models trained and saved")


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

    # ── Sentiment ──────────────────────────────────────────────────────
    try:
        res = requests.get(f"http://127.0.0.1:8001/sentiment?symbol={symbol}", timeout=2)
        sentiment_score      = res.json().get("score", 0)
        sentiment_score_norm = round((sentiment_score + 1) * 50, 2)
    except:
        sentiment_score = 0; sentiment_score_norm = 50

    # ── Data ───────────────────────────────────────────────────────────
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

    # ── Davey indicators ───────────────────────────────────────────────
    adx            = float(latest.get("ADX", 25))
    bb_pos         = float(latest.get("BB_pos", 0.5))
    bb_upper       = float(latest.get("BB_upper", price * 1.02))
    bb_lower       = float(latest.get("BB_lower", price * 0.98))
    mom_short      = float(latest.get("momentum_short", 0))
    mom_long       = float(latest.get("momentum_long", 0))
    close_pct_rank = float(latest.get("close_pct_rank", 0.5))
    vol_ratio      = float(latest.get("vol_ratio", 1.0))
    vol_breakout       = bool(latest.get("vol_breakout", False))
    adx_rising         = bool(latest.get("ADX_rising", True))
    bearish_divergence = bool(latest.get("bearish_divergence", False))
    bullish_divergence = bool(latest.get("bullish_divergence", False))
    range_contracting  = bool(latest.get("range_contracting", False))
    bull_tail_bar      = bool(latest.get("bull_tail_bar", False))
    bear_tail_bar      = bool(latest.get("bear_tail_bar", False))

    mfi            = float(latest.get("MFI", 50))
    mfi_oversold   = mfi < 20
    mfi_overbought = mfi > 80

    stoch_k          = float(latest.get("stoch_k", 50))
    stoch_d          = float(latest.get("stoch_d", 50))
    stoch_oversold   = stoch_k < 20 and stoch_d < 20
    stoch_overbought = stoch_k > 80 and stoch_d > 80

    zscore            = float(latest.get("zscore", 0))
    zscore_oversold   = zscore < -2.0
    zscore_overbought = zscore > 2.0

    # ── O'Neil indicators ──────────────────────────────────────────────
    is_leader          = bool(latest.get("is_leader", False))
    # ── Minervini Trend Template ──────────────────────────────────────
    trend_template   = bool(latest.get("trend_template", False))
    trend_score      = int(latest.get("trend_score", 0))
    tt1 = bool(latest.get("tt1", False))
    tt2 = bool(latest.get("tt2", False))
    tt3 = bool(latest.get("tt3", False))
    tt4 = bool(latest.get("tt4", False))
    tt5 = bool(latest.get("tt5", False))
    tt6 = bool(latest.get("tt6", False))
    tt7 = bool(latest.get("tt7", False))
    tt8 = bool(latest.get("tt8", False))
    tt9 = bool(latest.get("tt9", False))

    # ── Weinstein Stage ───────────────────────────────────────────────
    weinstein_stage2 = bool(latest.get("weinstein_stage2", False))
    weinstein_stage4 = bool(latest.get("weinstein_stage4", False))

    # ── VCP ───────────────────────────────────────────────────────────
    vcp_setup        = bool(latest.get("vcp_setup", False))
    vcp_contracting  = bool(latest.get("vcp_contracting", False))
    vol_dry          = bool(latest.get("vol_dry", False))

    # ── MA values ─────────────────────────────────────────────────────
    ma150 = float(latest.get("MA150", ma50))
    ma200 = float(latest.get("MA200", ma50))
    accum_days         = float(latest.get("accum_days", 10))
    distrib_days       = float(latest.get("distrib_days", 10))
    institutional_buying = bool(latest.get("institutional_buying", False))
    distribution_warning = distrib_days >= 5
    distrib_count      = float(latest.get("distrib_count_25d", 0))
    follow_through     = bool(latest.get("follow_through", False))
    market_top_warning = distrib_count >= 5
    near_52w_high      = bool(latest.get("near_52w_high", True))
    pct_from_52w       = float(latest.get("pct_from_52w_high", -10))

    # ── O'Neil CANSLIM — C: Current Quarterly Earnings ────────────────
    try:
        ticker = yf.Ticker(symbol)
        income = ticker.quarterly_financials
        if income is not None and not income.empty and "Net Income" in income.index:
            net_income = income.loc["Net Income"].dropna()
            if len(net_income) >= 5:
                q_latest    = float(net_income.iloc[0])
                q_year_ago  = float(net_income.iloc[4])
                eps_growth  = ((q_latest - q_year_ago) / abs(q_year_ago) * 100) if q_year_ago != 0 else 0
                q_prev      = float(net_income.iloc[1])
                q_prev_yago = float(net_income.iloc[5]) if len(net_income) > 5 else q_year_ago
                eps_growth_prev  = ((q_prev - q_prev_yago) / abs(q_prev_yago) * 100) if q_prev_yago != 0 else 0
                eps_accelerating = eps_growth > eps_growth_prev and eps_growth > 25
                eps_decelerating = eps_growth < eps_growth_prev * 0.5 and eps_growth_prev > 20
            else:
                eps_growth = 0; eps_accelerating = False; eps_decelerating = False
        else:
            eps_growth = 0; eps_accelerating = False; eps_decelerating = False
    except:
        eps_growth = 0; eps_accelerating = False; eps_decelerating = False

    # ── O'Neil CANSLIM — A: Annual Earnings + ROE ─────────────────────
    try:
        annual = ticker.financials
        if annual is not None and not annual.empty and "Net Income" in annual.index:
            net_annual = annual.loc["Net Income"].dropna()
            if len(net_annual) >= 3:
                y0 = float(net_annual.iloc[0])
                y1 = float(net_annual.iloc[1])
                y2 = float(net_annual.iloc[2])
                annual_growth_ok   = (y0 > y1 > y2) and y0 > 0
                annual_growth_rate = ((y0 - y2) / abs(y2) * 50) if y2 != 0 else 0
            else:
                annual_growth_ok = False; annual_growth_rate = 0
        else:
            annual_growth_ok = False; annual_growth_rate = 0
        roe        = float(ticker.info.get("returnOnEquity", 0) or 0) * 100
        roe_strong = roe > 17
    except:
        annual_growth_ok = False; annual_growth_rate = 0
        roe = 0; roe_strong = False

    # ── ADX zones ─────────────────────────────────────────────────────
    adx_flat      = adx < 20
    adx_trending  = 20 <= adx <= 35
    adx_exhausted = adx > 40

    liquid = vol_ratio >= 0.5

    prev_close      = float(prev["Close"])
    atr_breakout_up = price > (prev_close + 1.0 * atr)
    atr_breakout_dn = price < (prev_close - 1.0 * atr)

    bb_oversold   = bb_pos < 0.15
    bb_overbought = bb_pos > 0.85

    three_amigos_buy  = adx > 25 and rsi < 50 and mom_short > 0 and mom_long < 0
    three_amigos_sell = adx > 25 and rsi > 50 and mom_short < 0 and mom_long > 0

    if len(df) >= 3:
        h1 = float(df["High"].iloc[-2]); h2 = float(df["High"].iloc[-3])
        l1 = float(df["Low"].iloc[-2]);  l2 = float(df["Low"].iloc[-3])
        pullback_buy  = (h2 > h1) and (price > h2)
        pullback_sell = (l2 < l1) and (price < l2)
    else:
        pullback_buy = pullback_sell = False

    if len(df) >= 11:
        c0=float(df["Close"].iloc[-1]); c1=float(df["Close"].iloc[-2])
        c2=float(df["Close"].iloc[-3]); c3=float(df["Close"].iloc[-4])
        highest_10       = float(df["High"].iloc[-11:-1].max())
        lowest_10        = float(df["Low"].iloc[-11:-1].min())
        new_high_confirm = c0 > highest_10 and c0 > c1 and c0 > c3 and c1 > c2
        new_low_confirm  = c0 < lowest_10  and c0 < c1 and c0 < c3 and c1 < c2
    else:
        new_high_confirm = new_low_confirm = False

    if len(df) >= 4:
        c0=float(df["Close"].iloc[-1]); c1=float(df["Close"].iloc[-2])
        c2=float(df["Close"].iloc[-3]); c3=float(df["Close"].iloc[-4])
        three_up_closes   = c0 > c1 > c2 > c3
        three_down_closes = c0 < c1 < c2 < c3
    else:
        three_up_closes = three_down_closes = False

    pct_exit_long = close_pct_rank < 0.5

    # ── Timed Exit + Profit Tracker — Davey Exit #3 + #8 + #11 ───────
    tracker      = position_tracker.get(symbol, {})
    bars_held    = tracker.get("bars_held", 0)
    entry_price  = tracker.get("entry_price", price)
    peak_profit  = tracker.get("peak_profit", 0.0)
    direction    = tracker.get("direction", "")

    current_profit = (price - entry_price) if direction == "BUY" else (entry_price - price) if direction == "SELL" else 0.0
    if current_profit > peak_profit:
        peak_profit = current_profit
        if symbol in position_tracker:
            position_tracker[symbol]["peak_profit"] = peak_profit

    if peak_profit <= 0:
        gave_back_too_much = False
    elif peak_profit < atr * 1.0:
        gave_back_too_much = current_profit < peak_profit * 0.40 and bars_held > 3
    elif peak_profit < atr * 2.0:
        gave_back_too_much = current_profit < peak_profit * 0.60 and bars_held > 3
    else:
        gave_back_too_much = current_profit < peak_profit * 0.80 and bars_held > 3

    timed_exit_triggered = bars_held >= 10

    recent_high = float(df["High"].rolling(20).max().iloc[-2])
    recent_low  = float(df["Low"].rolling(20).min().iloc[-2])
    is_breakout_up   = price > recent_high
    is_breakout_down = price < recent_low
    valid_breakout_up   = is_breakout_up   and price > float(prev["High"]) and rsi > 50
    valid_breakout_down = is_breakout_down and price < float(prev["Low"])  and rsi < 50

    if   ma20 > ma50 and volatility > 0.01: regime = "TREND_UP"
    elif ma20 < ma50 and volatility > 0.01: regime = "TREND_DOWN"
    else:                                   regime = "RANGE"

    # ── ML ─────────────────────────────────────────────────────────────
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

    # ── Signals ────────────────────────────────────────────────────────
    signals, score = [], 0

    # Basic
    if ma20 > ma50: signals.append("MA20 > MA50 (Bullish)"); score += 1
    else:           signals.append("MA20 < MA50 (Bearish)"); score -= 1
    if rsi < 30:    signals.append("RSI Oversold"); score += 2
    elif rsi > 70:  signals.append("RSI Overbought"); score -= 2
    else:           signals.append("RSI Neutral")
    if momentum > 0: signals.append("Momentum Positive"); score += 1
    else:            signals.append("Momentum Negative"); score -= 1

    # Davey signals
    if adx_flat:       signals.append(f"ADX Low {adx:.1f} — Breakout Setup [Davey #4]"); score += 1
    elif adx_trending: signals.append(f"ADX Trending {adx:.1f} — Momentum Zone [Davey #4]")
    elif adx_exhausted: signals.append(f"ADX Exhausted {adx:.1f} — Avoid [Davey #4]"); score -= 2
    if bb_oversold:     signals.append("Near Bollinger Lower Band [Davey #25]"); score += 1
    elif bb_overbought: signals.append("Near Bollinger Upper Band [Davey #25]"); score -= 1
    if three_amigos_buy:   signals.append("Three Amigos BUY [Davey #27]"); score += 2
    elif three_amigos_sell: signals.append("Three Amigos SELL [Davey #27]"); score -= 2
    if pullback_buy:    signals.append("Quick Pullback Continuation [Davey #32]"); score += 2
    elif pullback_sell: signals.append("Quick Pullback SELL [Davey #32]"); score -= 2
    if new_high_confirm: signals.append("New High 4/4 Momentum Confirmed [Davey #37]"); score += 2
    elif new_low_confirm: signals.append("New Low 4/4 Momentum Confirmed [Davey #37]"); score -= 2
    if atr_breakout_up:  signals.append("ATR Significant Breakout UP [Davey #5]"); score += 1
    elif atr_breakout_dn: signals.append("ATR Significant Breakout DOWN [Davey #5]"); score -= 1
    if not liquid:       signals.append("LOW LIQUIDITY — Volume below average"); score -= 1
    if range_contracting: signals.append("Range Contracting — Breakout Setup [Davey #41]"); score += 1
    if bull_tail_bar:    signals.append("Bull Tail Bar — Institutional Rejection [Davey #36]"); score += 2
    if bear_tail_bar:    signals.append("Bear Tail Bar — Distribution Signal [Davey #36]"); score -= 2
    if mfi_oversold:     signals.append(f"MFI Oversold {mfi:.1f} — Institutional Buying [Davey #23]"); score += 2
    elif mfi_overbought: signals.append(f"MFI Overbought {mfi:.1f} — Institutional Selling [Davey #23]"); score -= 2
    if stoch_oversold:   signals.append(f"Stochastic Oversold {stoch_k:.1f} [Davey #24]"); score += 1
    elif stoch_overbought: signals.append(f"Stochastic Overbought {stoch_k:.1f} [Davey #24]"); score -= 1
    if vol_breakout:     signals.append("Volume Breakout 1.5x [Davey #1]"); score += 1
    if adx_rising:       signals.append("ADX Rising — Trend Strengthening [Davey #8]"); score += 1
    if bullish_divergence: signals.append("Bullish RSI Divergence [Davey #12]"); score += 2
    if bearish_divergence: signals.append("Bearish RSI Divergence [Davey #12]"); score -= 2
    if zscore_oversold:   signals.append(f"Z-Score Oversold {zscore:.2f} [Aldridge]"); score += 2
    elif zscore_overbought: signals.append(f"Z-Score Overbought {zscore:.2f} [Aldridge]"); score -= 2

    # O'Neil signals
    if eps_accelerating:   signals.append(f"EPS Accelerating +{eps_growth:.0f}% YoY [O'Neil C]"); score += 3
    if eps_decelerating:   signals.append("EPS Decelerating — Caution [O'Neil C]"); score -= 2
    if annual_growth_ok:   signals.append("3 Years Consecutive Earnings Growth [O'Neil A]"); score += 2
    if roe_strong:         signals.append(f"ROE {roe:.1f}% > 17% [O'Neil A]"); score += 1
    if institutional_buying: signals.append(f"Accumulation {accum_days:.0f} > Distribution {distrib_days:.0f} [O'Neil I]"); score += 2
    if distribution_warning: signals.append(f"⚠️ {distrib_days:.0f} Distribution Days [O'Neil M]"); score -= 3
    if market_top_warning: signals.append("⛔ 5+ Distribution Days — Market Top Warning [O'Neil M]"); score -= 3
    if follow_through:     signals.append("Follow-Through Day — Uptrend Confirmed [O'Neil M]"); score += 2
    if near_52w_high:      signals.append(f"Near 52W High ({pct_from_52w:.1f}%) [O'Neil N]"); score += 1
    else:                  signals.append(f"Far from 52W High ({pct_from_52w:.1f}%) — Avoid [O'Neil]"); score -= 1
    if is_leader:          signals.append("Outperforming Market — Leader [O'Neil L]"); score += 1
    # Minervini Trend Template signals
    if trend_template:
        signals.append(f"✅ Minervini Trend Template — ALL 8 CONDITIONS MET [{trend_score}/9]"); score += 4
    elif trend_score >= 6:
        signals.append(f"Trend Template Partial — {trend_score}/9 conditions [Minervini]"); score += 2
    elif trend_score >= 4:
        signals.append(f"Trend Template Weak — {trend_score}/9 conditions [Minervini]"); score += 0
    else:
        signals.append(f"❌ Trend Template FAILED — {trend_score}/9 [Minervini] — Avoid"); score -= 3

    # Weinstein Stage
    if weinstein_stage2:
        signals.append("Stage 2 Uptrend — Weinstein Confirmed"); score += 2
    if weinstein_stage4:
        signals.append("⛔ Stage 4 Decline — Weinstein — Never Buy"); score -= 4

    # VCP
    if vcp_setup:
        signals.append("VCP Setup — Volatility Contracting + Volume Drying [Minervini]"); score += 3
    elif vcp_contracting:
        signals.append("Range Contracting — Partial VCP [Minervini]"); score += 1
    # Exit signals
    if three_up_closes:      signals.append("⚠️ 3 Consecutive Up Closes — Exit Signal [Davey #6]")
    if three_down_closes:    signals.append("⚠️ 3 Consecutive Down Closes — Exit Signal [Davey #6]")
    if timed_exit_triggered: signals.append(f"⚠️ Held {bars_held} days — Timed Exit [Davey #3]")
    if gave_back_too_much:   signals.append("⚠️ Gave back peak profit — Exit [Davey #8/#11]")

    technical_score = round(((score + 12) / 24) * 100, 2)
    confidence = round(0.5 * ml_confidence + 0.3 * technical_score + 0.2 * sentiment_score_norm, 2)

    # ═══════════════════════════════════════════════════════════════════
    # DECISION ENGINE
    # ═══════════════════════════════════════════════════════════════════
    prediction = "HOLD"

    # L1: Timed exit
    if timed_exit_triggered:
        prediction = "HOLD"
        position_tracker.pop(symbol, None)

    # L2: ADX exhausted — no new entries
    elif adx_exhausted:
        prediction = "HOLD"

    # L3: Weinstein Stage 4 — never buy declining stocks
    elif weinstein_stage4:
        prediction = "HOLD"

    # L4: Minervini Trend Template — need at least 6/9 for any buy signal
    elif trend_score < 6 and regime == "TREND_UP":
        prediction = "HOLD"

    # L5: Liquidity — skip illiquid stocks
    elif not liquid:
        prediction = "HOLD"

    else:
        # ── TREND_UP ──────────────────────────────────────────────────
        if regime == "TREND_UP":
            if market_top_warning:
                prediction = "HOLD"
            else:
                buy = False
                if valid_breakout_up and (adx_flat or adx_trending) and atr_breakout_up and vol_breakout and range_contracting: buy = True
                if three_amigos_buy and score_ml >= 0.5 and adx_rising: buy = True
                if pullback_buy and score_ml >= 0.5 and (adx_flat or adx_trending): buy = True
                if new_high_confirm and score_ml >= 0.6 and vol_breakout: buy = True
                if bullish_divergence and score_ml >= 0.5: buy = True
                if bull_tail_bar and score_ml >= 0.5: buy = True
                # G: VCP breakout — Minervini's highest probability setup
                if vcp_setup and vol_breakout and trend_template: buy = True
                if buy: prediction = "BUY"

        # ── TREND_DOWN ────────────────────────────────────────────────
        elif regime == "TREND_DOWN":
            sell = False
            if market_top_warning: sell = True
            if valid_breakout_down and (adx_flat or adx_trending) and atr_breakout_dn and vol_breakout: sell = True
            if three_amigos_sell and score_ml <= 0.5 and adx_rising: sell = True
            if pullback_sell and score_ml <= 0.5 and (adx_flat or adx_trending): sell = True
            if new_low_confirm and score_ml <= 0.4 and vol_breakout: sell = True
            if bearish_divergence and score_ml <= 0.5: sell = True
            if sell: prediction = "SELL"

        # ── RANGE ─────────────────────────────────────────────────────
        elif regime == "RANGE":
            if (rsi < 30 or bb_oversold) and mfi_oversold:        prediction = "BUY"
            elif (rsi > 70 or bb_overbought) and mfi_overbought:   prediction = "SELL"
            elif rsi < 30 and bb_oversold:                         prediction = "BUY"
            elif rsi > 70 and bb_overbought:                       prediction = "SELL"
            elif rsi < 30:                                         prediction = "BUY"
            elif rsi > 70:                                         prediction = "SELL"
            elif zscore_oversold and rsi < 40:                     prediction = "BUY"
            elif zscore_overbought and rsi > 60:                   prediction = "SELL"

        # ── Exit overrides ────────────────────────────────────────────
        if prediction == "BUY"  and three_down_closes:                   prediction = "HOLD"
        if prediction == "SELL" and three_up_closes:                      prediction = "HOLD"
        if prediction == "BUY"  and pct_exit_long and bars_held > 5:     prediction = "HOLD"
        if gave_back_too_much:                                            prediction = "HOLD"
        if prediction == "BUY"  and bearish_divergence:                   prediction = "HOLD"
        if prediction == "SELL" and bullish_divergence:                   prediction = "HOLD"

    # ── Standard filters ──────────────────────────────────────────────
    if confidence < 52:    prediction = "HOLD"
    if volatility < 0.005: prediction = "HOLD"

    # ── O'Neil 8% Hard Stop ───────────────────────────────────────────
    if symbol in position_tracker:
        ep  = position_tracker[symbol].get("entry_price", price)
        dir_p = position_tracker[symbol].get("direction", "")
        if dir_p == "BUY"  and (ep - price) / ep * 100 >= 8:
            prediction = "HOLD"
            signals.append("⛔ 8% Hard Stop — O'Neil Rule")
            position_tracker.pop(symbol, None)
        elif dir_p == "SELL" and (price - ep) / ep * 100 >= 8:
            prediction = "HOLD"
            signals.append("⛔ 8% Hard Stop — O'Neil Rule")
            position_tracker.pop(symbol, None)

    if prediction == "BUY"  and sentiment_score < -0.2: prediction = "HOLD"
    if prediction == "SELL" and sentiment_score >  0.2: prediction = "HOLD"

    # ── Serial Correlation — Davey Entry #16 ─────────────────────────
    last_trade_data = position_tracker.get("_last_trade", {})
    last_trade_date = last_trade_data.get("date", None)
    if last_trade_date:
        days_since = (datetime.now() - last_trade_date).days
        wait_days  = 5 if last_trade_data.get("result","none") == "loss" else 2
        if days_since < wait_days:
            prediction = "HOLD"

    # ── Update tracker ────────────────────────────────────────────────
    if prediction in ("BUY","SELL"):
        last_trade_time = datetime.now()
        position_tracker[symbol] = {
            "bars_held": 0, "entry_price": price,
            "direction": prediction, "peak_profit": 0.0,
        }
        position_tracker["_last_trade"] = {
            "date": datetime.now(), "result": "win", "symbol": symbol,
        }
    elif symbol in position_tracker:
        position_tracker[symbol]["bars_held"] += 1

    # ── Trade plan ────────────────────────────────────────────────────
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
        risk_pu       = abs(entry - stop_loss)
        position_size = int((capital * risk_per_trade) / risk_pu) if risk_pu > 0 else 0
        capital_used  = round(position_size * entry, 2)
        rr            = round(abs(target - entry) / abs(entry - stop_loss), 2) if abs(entry - stop_loss) else 0
    else:
        position_size = capital_used = 0; rr = "-"

    if prediction in ("BUY","SELL"):
        trade = {"time": str(datetime.now()), "symbol": symbol, "prediction": prediction,
                 "price": round(price,2), "confidence": confidence, "regime": regime,
                 "target": target, "stop_loss": stop_loss, "position_size": position_size}
        paper_trades.append(trade); open_positions.append(trade)

    # ── AI analysis ───────────────────────────────────────────────────
    trend  = "Uptrend" if ma20 > ma50 else "Downtrend"
    reason = []
    if ma20 > ma50:      reason.append("Bullish trend")
    else:                reason.append("Bearish trend")
    if rsi < 30:         reason.append("RSI oversold")
    elif rsi > 70:       reason.append("RSI overbought")
    if momentum > 0:     reason.append("Positive momentum")
    if three_amigos_buy: reason.append("Three Amigos BUY [Davey]")
    if pullback_buy:     reason.append("Quick Pullback [Davey]")
    if new_high_confirm: reason.append("New High 4/4 [Davey]")
    if eps_accelerating: reason.append(f"EPS +{eps_growth:.0f}% [O'Neil]")
    if annual_growth_ok: reason.append("3Y earnings growth [O'Neil]")
    if not liquid:       reason.append("LOW LIQUIDITY")

    ai_analysis  = analyze_stock_ai(symbol, trend, round(rsi,2), prediction, confidence)
    ai_analysis += "\n\nReasons:\n" + ", ".join(reason)

    info       = get_stock_info(symbol)
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
        "fundamental": {
            "eps_growth":        round(eps_growth, 2),
            "eps_accelerating":  eps_accelerating,
            "eps_decelerating":  eps_decelerating,
            "annual_growth_ok":  annual_growth_ok,
            "roe":               round(roe, 2),
            "roe_strong":        roe_strong,
        },
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
            "near_52w_high": near_52w_high, "pct_from_52w": round(pct_from_52w,2),
            "market_top_warning": market_top_warning, "follow_through": follow_through,
            "institutional_buying": institutional_buying,
            "trailing_stop": round(price-(2.0*atr),2) if prediction=="BUY" else round(price+(2.0*atr),2) if prediction=="SELL" else "-",
            "trend_template": trend_template,
            "trend_score": f"{trend_score}/9",
            "weinstein_stage2": weinstein_stage2,
            "weinstein_stage4": weinstein_stage4,
            "vcp_setup": vcp_setup,
            "ma150": round(ma150, 2),
            "ma200": round(ma200, 2),
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
    top5       = results[:5]
    total_conf = sum(s["confidence"] for s in top5) or 1
    capital    = 10000
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
        f    = scaler.transform(feat)
        sc   = (dt_model.predict(f)[0]*model_weights["dt"] +
                rf_model.predict(f)[0]*model_weights["rf"] +
                lr_model.predict(f)[0]*model_weights["lr"] +
                xgb_model.predict(f)[0]*model_weights["xgb"])
        pred = 1 if sc >= 0.5 else 0
        cur  = float(row["Close"]); nxt = float(df.iloc[i+1]["Close"])
        if pred == 1: wins += 1 if nxt > cur else 0; losses += 0 if nxt > cur else 1
        else:         wins += 1 if nxt < cur else 0; losses += 0 if nxt < cur else 1
        slip = 0.002
        if pred == 1:
            change = (nxt*(1-slip) - cur*(1+slip)) / (cur*(1+slip))
        else:
            change = (cur*(1-slip) - nxt*(1+slip)) / (cur*(1-slip))
        profit += 1000 * change
    total = wins + losses
    sharpe_est = round(profit / (1000 * total) / 0.02, 2) if total else 0
    return {"symbol": symbol, "accuracy": round(wins/total*100,2) if total else 0,
            "total_trades": total, "wins": wins, "losses": losses,
            "profit": round(profit,2), "final_capital": round(10000+profit,2),
            "sharpe_benchmark": f"Target > 1.0 (yours: {sharpe_est})"}


@app.get("/simulate-profit")
def simulate_profit(symbol: str = "AAPL"):
    df = get_stock_data(symbol); df = prepare_features(df)
    capital = 10000
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
        f    = scaler.transform(feat)
        pred = rf_model.predict(f)[0]
        nxt  = float(df.iloc[i+1]["Close"]); cur = float(row["Close"])
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
        row  = df.iloc[i].copy()
        feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
        f    = scaler.transform(feat)
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
                row  = df.iloc[-2]
                feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
                f    = scaler.transform(feat)
                sc   = (dt_model.predict(f)[0]*model_weights["dt"] +
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
                    row  = df.iloc[step]
                    feat = pd.DataFrame([row[["RSI","MA20","MA50","momentum","volatility"]]])
                    f    = scaler.transform(feat)
                    sc   = (dt_model.predict(f)[0]*model_weights["dt"] +
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
    log_experiment("v3_canslim",f"mode={mode}",{
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