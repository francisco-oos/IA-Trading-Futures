import pandas as pd

df = pd.read_csv("data/ohlcv_combined.csv")
print(df.columns.tolist())
