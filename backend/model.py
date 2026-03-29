from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def prepare_data(df):
    df["return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["RSI"] = calculate_rsi(df)

    df = df.dropna()

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna()

    X = df[["return", "MA5", "MA20", "RSI"]]
    X = df[["return", "MA5", "MA20"]]
    y = df["target"]

    return X, y

def calculate_rsi(df, window=14):
    delta = df["Close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def train_models(X, y):
    model1 = LogisticRegression()
    model2 = RandomForestClassifier()

    model1.fit(X, y)
    model2.fit(X, y)

    return model1, model2