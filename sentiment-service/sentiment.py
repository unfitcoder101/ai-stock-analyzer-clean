import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from functools import lru_cache

analyzer = SentimentIntensityAnalyzer()


def get_news(symbol):
    stock = yf.Ticker(symbol)
    news = stock.news

    headlines = []

    company_map = {
        "AAPL": "apple",
        "MSFT": "microsoft",
        "TSLA": "tesla",
        "NVDA": "nvidia",
        "RELIANCE.NS": "reliance",
        "TCS.NS": "tata consultancy",
    }

    company_name = company_map.get(symbol, symbol).lower()
    clean_symbol = symbol.lower().split(".")[0]

    if news:
        for item in news[:10]:
            content = item.get("content", {})
            title = content.get("title")
            source = content.get("provider", {}).get("displayName", "unknown")

            if not title or not isinstance(title, str):
                continue

            title = title.strip()
            title_lower = title.lower()

            # ✅ FIXED position
            if any(word in title_lower for word in ["should you", "top 5", "best stocks"]):
                continue

            if any(word in title_lower for word in ["prediction", "target", "forecast"]):
                continue

            if (
                company_name in title_lower
                or clean_symbol in title_lower
            ):
                headlines.append({
                    "title": title,
                    "source": source
                })

            if len(headlines) >= 5:
                break

    return headlines

def analyze_sentiment(headlines):
    source_weights = {
        "Reuters": 1.0,
        "Bloomberg": 1.0,
        "Yahoo Finance": 0.8,
        "Yahoo Finance Video": 0.7,
        "Motley Fool": 0.5,
        "Simply Wall St.": 0.6,
        "Trefis": 0.5,
        "MT Newswires": 0.7,
        "unknown": 0.5
    }

    scores = []

    for item in headlines:
        text = item["title"]
        source = item["source"]

        if not text:
            continue

        text_lower = text.lower()

        weight = source_weights.get(source, 0.5)

        # 🔥 NEW: reduce opinion/noise impact
        if any(word in text_lower for word in ["buy", "sell", "should you"]):
            weight *= 0.5

        score = analyzer.polarity_scores(text)["compound"]

        scores.append(score * weight)

    if not scores:
        return 0

    # recency weighting (assumes sorted news)
    weights = [1.0, 0.8, 0.6, 0.4, 0.2]
    weights = weights[:len(scores)]

    weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    return weighted_score


@lru_cache(maxsize=100) #Add TTL manually in production
def get_company_sentiment(symbol):
    headlines = get_news(symbol)
    score = analyze_sentiment(headlines)

    return score, headlines