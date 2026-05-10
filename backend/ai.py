import os 
from groq import Groq

def analyze_stock_ai(symbol, trend, rsi, prediction, confidence):
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"""You are a sharp equity analyst. Give a 3-sentence analysis:

Stock: {symbol}
Trend: {trend}
RSI: {rsi}
Signal: {prediction}
Confidence: {confidence}%

Be specific. Mention one risk. End with a clear action."""
            }],
            max_tokens=200
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Analysis unavailable: {str(e)}"