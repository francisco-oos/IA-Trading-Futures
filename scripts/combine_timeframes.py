import pandas as pd
import os

def safe_read_csv(path, timeframe):
    if not os.path.exists(path):
        print(f"Archivo no encontrado: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.rename(columns=lambda col: f"{col}_{timeframe}" if col != "timestamp" else "timestamp")
        return df
    except Exception as e:
        print(f"Error al leer {path}: {e}")
        return pd.DataFrame()

def combine_timeframes():
    os.makedirs("data", exist_ok=True)

    df_1m = safe_read_csv("data/ohlcv_1m.csv", "1m")
    df_5m = safe_read_csv("data/ohlcv_5m.csv", "5m")
    df_15m = safe_read_csv("data/ohlcv_15m.csv", "15m")

    if df_1m.empty:
        print("No hay datos de 1m, no se puede combinar.")
        return

    # Combinar usando timestamp
    df_combined = df_1m.copy()
    if not df_5m.empty:
        df_combined = df_combined.merge(df_5m, on="timestamp", how="left")
    if not df_15m.empty:
        df_combined = df_combined.merge(df_15m, on="timestamp", how="left")

    # Eliminar duplicados
    df_combined.drop_duplicates(subset="timestamp", inplace=True)

    # Ordenar
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

    # Guardar
    output_path = "data/ohlcv_combined.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"Datos combinados correctamente: {output_path}")

if __name__ == "__main__":
    combine_timeframes()
