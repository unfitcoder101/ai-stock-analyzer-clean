from fastapi import FastAPI
from sentiment import get_company_sentiment

app = FastAPI()

@app.get("/sentiment")
def sentiment(symbol: str):
    score, headlines = get_company_sentiment(symbol.upper())

    # ✅ ADD THIS HERE
    if len(headlines) == 0:
        return {
            "symbol": symbol.upper(),
            "score": 0,
            "label": "neutral",
            "reason": "insufficient data",
            "headlines": headlines
        }

    # existing logic
    if score > 0.2:
        label = "positive"
    elif score < -0.2:
        label = "negative"
    else:
        label = "neutral"

    return {
        "symbol": symbol.upper(),
        "score": round(score, 3),
        "label": label,
        "headlines": headlines
    }