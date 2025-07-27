import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Parámetros
future_steps = 3  # cuántos pasos hacia adelante para predecir

# Cargar datos combinados
df = pd.read_csv("data/ohlcv_combined.csv", parse_dates=["timestamp"])

# Crear columna objetivo: 1 si cierre futuro > cierre actual, 0 si no
df["future_close"] = df["close_1m"].shift(-future_steps)
df["target"] = (df["future_close"] > df["close_1m"]).astype(int)

# Eliminar filas con NaN (últimas filas donde no hay 'future_close')
df.dropna(inplace=True)

# Seleccionar features (todas las columnas menos timestamp, future_close y target)
features = [col for col in df.columns if col not in ["timestamp", "future_close", "target"]]

X = df[features]
y = df["target"]

# Dividir datos: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Crear modelo XGBoost (clasificación)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

# Entrenar modelo
model.fit(X_train, y_train)

# Predecir y evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")

# Guardar modelo
joblib.dump(model, "models/xgb_model.joblib")
print("Modelo guardado en models/xgb_model.joblib")
