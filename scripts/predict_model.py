import pandas as pd
import joblib
import sys
import os

def load_model(path="models/xgb_model_robust.joblib"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo no encontrado en {path}")
    model = joblib.load(path)
    return model

def predict(df, model):
    # Asumimos que df tiene las mismas columnas/features usados en entrenamiento
    # Quitar columnas no usadas (timestamp, target, future_close)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'future_close', 'target']]
    X = df[feature_cols]
    preds = model.predict(X)
    # Opcional: probs = model.predict_proba(X)
    df['prediction'] = preds
    return df

def main(file_path):
    print(f"Cargando datos desde {file_path}...")
    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    print(f"Datos cargados con {len(df)} filas.")

    model = load_model()
    print("Modelo cargado correctamente.")

    df_pred = predict(df, model)
    print("Predicciones generadas:")
    print(df_pred[['timestamp', 'prediction']].tail(10))

    # Guardar resultados con predicciones
    output_path = "data/predictions.csv"
    df_pred.to_csv(output_path, index=False)
    print(f"Archivo con predicciones guardado en {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python scripts/predict_model.py <archivo_csv_con_datos>")
        sys.exit(1)
    input_file = sys.argv[1]
    main(input_file)
