# download_data.py

import os
import time
import pandas as pd
from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, USE_TESTNET

# Inicializar cliente de Binance
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
if USE_TESTNET:
    client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

# Descargar datos históricos
def download_ohlcv(symbol, interval, lookback_minutes):
    limit = 1500
    interval_mapping = {
        '1m': 1,
        '5m': 5,
        '15m': 15
    }
    interval_minutes = interval_mapping[interval]
    total_bars = lookback_minutes // interval_minutes
    all_klines = []

    print(f"📥 Descargando {interval} desde {lookback_minutes} minutos atrás...")

    last_timestamp = None

    while total_bars > 0:
        candles = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=min(limit, total_bars),
            endTime=last_timestamp
        )
        if not candles:
            break

        all_klines = candles + all_klines
        last_timestamp = candles[0][0] - 1
        total_bars -= len(candles)
        time.sleep(0.2)

    # Convertir a DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    # Guardar archivo CSV
    os.makedirs("data", exist_ok=True)
    filename = f"data/ohlcv_{interval}.csv"
    df.to_csv(filename)
    print(f"✅ Guardado en {filename}")

    return df

if __name__ == "__main__":
    INTERVALS = ['1m', '5m', '15m']
    LOOKBACK_MINUTES = 3 * 24 * 60  # 3 días

    for interval in INTERVALS:
        download_ohlcv(SYMBOL, interval, LOOKBACK_MINUTES)
