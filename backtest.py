# backtest.py
import pandas as pd
from strategy import generate_signal

df = pd.read_csv('data/btc_futures_ohlcv.csv', index_col='timestamp', parse_dates=True)

signals = []
for i in range(50, len(df)):
    slice_df = df.iloc[:i].copy()
    signal = generate_signal(slice_df)
    signals.append(signal)

print(signals[-10:])