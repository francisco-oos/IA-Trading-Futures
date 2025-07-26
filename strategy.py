# strategy.py
import pandas as pd
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD

def generate_signal(df):
    model = joblib.load('model/xgboost_model.pkl')
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd()
    df.dropna(inplace=True)

    features = df[['rsi', 'macd']].iloc[-1:]
    pred = model.predict(features)[0]
    return 'buy' if pred == 1 else 'sell'