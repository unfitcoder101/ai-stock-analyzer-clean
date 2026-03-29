from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



def analyze_stock_ai(symbol, trend, rsi, prediction, confidence):
    prompt = f"""
You are a professional stock analyst.

Symbol: {symbol}
Trend: {trend}
RSI: {rsi}
Prediction: {prediction}
Confidence: {confidence}%

Give:
1. Short summary
2. Bull case
3. Bear case
4. Final verdict
5. Trade advice
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content