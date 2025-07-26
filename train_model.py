# train_model.py
import pandas as pd
import xgboost as xgb
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('data/btc_futures_ohlcv.csv', index_col='timestamp', parse_dates=True)
df['return'] = df['close'].pct_change()
df['rsi'] = RSIIndicator(df['close']).rsi()
df['macd'] = MACD(df['close']).macd()
df.dropna(inplace=True)

df['target'] = df['return'].shift(-1)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

features = ['rsi', 'macd']
X = df[features]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBClassifier()
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Accuracy: {acc:.2f}")

joblib.dump(model, 'model/xgboost_model.pkl')