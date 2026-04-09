import yfinance as yf
import pandas as pd

def get_stock_data(symbol):
    import time
    import random

    time.sleep(random.uniform(0.1, 0.3))

    try:
        df = yf.download(symbol, period="1y", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df.dropna(inplace=True)
        return df
    except:
        return pd.DataFrame()

def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            "name": info.get("longName", symbol),
            "sector": info.get("sector"),
            "price": info.get("currentPrice"),
        }
    except:
        return {
            "name": symbol,
            "sector": "Unknown",
            "price": None
        }
