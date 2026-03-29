import yfinance as yf
import pandas as pd

def get_stock_data(symbol="AAPL"):
    df = yf.download(symbol, period="3mo")
    df = df.dropna()
    return df