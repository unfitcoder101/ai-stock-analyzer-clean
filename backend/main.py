from fastapi import FastAPI
from ai import analyze_stock_ai
from data import get_stock_data
from model import prepare_data, train_models
from fastapi.middleware.cors import CORSMiddleware


try:
    ai_analysis = analyze_stock_ai(...)
except Exception as e:
    ai_analysis = f"AI error: {str(e)}"

history = []
watchlist = []

def backtest(df):
    correct = 0
    total = 0

    X, y = prepare_data(df)

    for i in range(len(X) - 1):
        sample = X.iloc[[i]]

        pred1 = model1.predict(sample)[0]
        pred2 = model2.predict(sample)[0]

        final = 1 if [pred1, pred2].count(1) > 1 else 0

        actual = y.iloc[i]

        if final == actual:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0

    return round(accuracy * 100, 2)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model1 = None
model2 = None


@app.on_event("startup")
def load_models():
    global model1, model2

    df = get_stock_data("AAPL")
    X, y = prepare_data(df)

    model1, model2 = train_models(X, y)

@app.get("/best-stock")
def best_stock():
    symbols = [
    "AAPL", "MSFT", "GOOGL", "TSLA",
    "AMZN", "META",
    "RELIANCE.NS", "TCS.NS", "INFY.NS"
]

    best = None
    best_score = -999

    for symbol in symbols:
        try:
            df = get_stock_data(symbol)
            X, _ = prepare_data(df)

            latest = X.iloc[[-1]]

            pred1 = model1.predict(latest)[0]
            pred2 = model2.predict(latest)[0]

            final = 1 if [pred1, pred2].count(1) > 1 else 0

            rsi = df["RSI"].iloc[-1]

            score = 0

            # ML
            score += 1 if final == 1 else -1

            # RSI
            if rsi < 30:
                score += 1
            elif rsi > 70:
                score -= 1

            if score > best_score:
                best_score = score
                best = symbol

        except:
            continue

    return {
        "best_stock": best,
        "score": best_score
    }

@app.get("/top-stocks")
def top_stocks():
    symbols = [
    "AAPL", "MSFT", "GOOGL", "TSLA",
    "AMZN", "META",
    "RELIANCE.NS", "TCS.NS", "INFY.NS"
]

    results = []

    for symbol in symbols:
        try:
            df = get_stock_data(symbol)
            X, _ = prepare_data(df)

            latest = X.iloc[[-1]]

            pred1 = model1.predict(latest)[0]
            pred2 = model2.predict(latest)[0]

            final = 1 if [pred1, pred2].count(1) > 1 else 0

            results.append({
                "symbol": symbol,
                "prediction": "BUY" if final == 1 else "SELL"
            })

        except:
            continue

    return results
@app.get("/history")
def get_history():
    return history

@app.get("/add-watchlist")
def add_watchlist(symbol: str):
    if symbol not in watchlist:
        watchlist.append(symbol)
    return {"watchlist": watchlist}


@app.get("/watchlist")
def get_watchlist():
    return watchlist

@app.get("/popular-stocks")
def popular_stocks():
    return [
        "AAPL", "MSFT", "GOOGL", "TSLA",
        "AMZN", "META",
        "RELIANCE.NS", "TCS.NS", "INFY.NS",
        "BTC-USD"
    ]

@app.get("/")
def home():
    return {"message": "ML API running"}

from fastapi import Query

from fastapi import Query

@app.get("/predict")
def predict(symbol: str = Query("AAPL")):
    df = get_stock_data(symbol)

    
    # STEP 1 — validation
    if df.empty or "Close" not in df.columns:
        return {"error": "Invalid symbol or no data found"}

    if len(df) < 20:
        return {"error": "Not enough data"}

    # STEP 2 — prepare data
    X, _ = prepare_data(df)

    # STEP 3 — trend
    ma5 = df["Close"].rolling(5).mean()
    ma20 = df["Close"].rolling(20).mean()

    ma5_last = float(ma5.iloc[-1].item())
    ma20_last = float(ma20.iloc[-1].item())

    trend = "Uptrend" if ma5_last > ma20_last else "Downtrend"

    # STEP 4 — ML predictions
    latest = X.iloc[[-1]]

    pred1 = model1.predict(latest)[0]
    pred2 = model2.predict(latest)[0]

    votes = [pred1, pred2]
    buy_votes = votes.count(1)
    sell_votes = votes.count(0)

    final = 1 if buy_votes > sell_votes else 0
    confidence = max(buy_votes, sell_votes) / len(votes)

    # STEP 5 — RSI
    rsi_value = df["RSI"].iloc[-1]

    signal = "HOLD"
    if rsi_value < 30:
        signal = "BUY (Oversold)"
    elif rsi_value > 70:
        signal = "SELL (Overbought)"

    # STEP 6 — SCORE SYSTEM
    score = 0

    if trend == "Uptrend":
        score += 1
    else:
        score -= 1

    if rsi_value < 30:
        score += 1
    elif rsi_value > 70:
        score -= 1

    if final == 1:
        score += 1
    else:
        score -= 1

    bullish = 0
    bearish = 0

    # Trend
    if trend == "Uptrend":
        bullish += 1
    else:
        bearish += 1

    # RSI
    if rsi_value < 30:
        bullish += 1
    elif rsi_value > 70:
        bearish += 1

    # ML
    if final == 1:
        bullish += 1
    else:
        bearish += 1

    # RESULT
    if bullish > bearish:
        confluence = "Bullish"
    elif bearish > bullish:
        confluence = "Bearish"
    else:
        confluence = "Neutral"

    entry = "Wait"
    stop_loss = "N/A"
    target = "N/A"

    insight = ""

    if confluence == "Bullish" and rsi_value < 40:
        entry = "Good Buy Zone"
        stop_loss = "2% below"
        target = "5% above"

    elif confluence == "Bearish" and rsi_value > 60:
        entry = "Avoid / Sell Zone"
        stop_loss = "N/A"
        target = "N/A"

    

    if confluence == "Bullish":
        insight = "Stock shows bullish signals with decent momentum. Possible buying opportunity."

    elif confluence == "Bearish":
      insight = "Stock is in a downtrend with weak signals. Avoid entering now."

    else:
        insight = "Mixed signals. Wait for clearer confirmation."

     # STEP 9 — EXPLANATION
    positive = []
    negative = []

    if trend == "Uptrend":
        positive.append("Uptrend detected")
    else:
        negative.append("Downtrend detected")

    if rsi_value < 30:
        positive.append("Stock is oversold")
    elif rsi_value > 70:
        negative.append("Stock is overbought")
    else:
        negative.append("RSI is neutral")

    if final == 1:
        positive.append("Models suggest BUY")
    else:
        negative.append("Models suggest SELL")

    backtest_accuracy = backtest(df)

    # STEP 7 — DECISION STRENGTH
    if confidence > 0.8:
        strength = "Strong"
    elif confidence > 0.6:
        strength = "Moderate"
    else:
        strength = "Weak"

    # STEP 8 — MARKET STATE
    volatility = float(df["Close"].pct_change().std())

    if volatility > 0.03:
        market_state = "Volatile"
    elif abs(ma5_last - ma20_last) < 1:
        market_state = "Sideways"
    else:
        market_state = "Trending"

   
    
    history.append({
        "symbol": symbol,
        "prediction": "BUY" if final == 1 else "SELL",
        "confidence": round(confidence * 100, 2)    
    })

# keep only last 10
    if len(history) > 10:
        history.pop(0)
    

    ai_analysis = analyze_stock_ai(
    symbol,
    trend,
    rsi_value,
    "BUY" if final == 1 else "SELL",
    round(confidence * 100, 2)
)
    # FINAL RESPONSE
    return {
        "prediction": "BUY" if final == 1 else "SELL",
        "confidence": round(confidence * 100, 2),
        "decision_strength": strength,

        "trend": trend,
        "market_state": market_state,

        "rsi": round(rsi_value, 2),
        "signal": signal,

        "score": score,
        "backtest_accuracy": backtest_accuracy,
        "models": {
            "logistic": "BUY" if pred1 == 1 else "SELL",
            "random_forest": "BUY" if pred2 == 1 else "SELL"
        },

        "reasoning": {
            "positive": positive,
            "negative": negative
        },

        "confluence": confluence,

        "trade_plan": {
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target
        },

        "insight": insight,
        "ai_analysis": ai_analysis

    }