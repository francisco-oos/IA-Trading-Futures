import sys
import os
import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from datetime import datetime, timedelta, timezone

# Ruta del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import BINANCE_API_KEY, BINANCE_API_SECRET, SYMBOL, USE_TESTNET

# Cliente Binance
client = Client(BINANCE_API_KEY, BINANCE_API_SECRET, testnet=USE_TESTNET)
symbol = SYMBOL

def download_ohlcv_incremental(symbol, interval, lookback_days=30, chunk_minutes=60):
    """
    Descarga datos OHLCV por chunks de tiempo para evitar timeouts.
    chunk_minutes: tamaño del chunk (en minutos) para cada request (p.ej. 60 min)
    """
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=lookback_days)

    print(f"Descargando datos de {interval} desde {start_time.strftime('%d %b %Y')} hasta {end_time.strftime('%d %b %Y')}")

    all_data = []
    limit = 1000
    delta = timedelta(minutes=chunk_minutes)
    max_retries = 5

    current_start = start_time

    while current_start < end_time:
        current_end = min(current_start + delta, end_time)
        start_str = current_start.strftime('%d %b %Y %H:%M:%S')
        end_str = current_end.strftime('%d %b %Y %H:%M:%S')

        retries = 0
        success = False

        while not success and retries < max_retries:
            try:
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_str,
                    end_str=end_str,
                    limit=limit,
                    requests_params={"timeout": 20}
                )
                success = True
            except (BinanceAPIException, BinanceRequestException) as e:
                retries += 1
                print(f"Error de API: {e}. Reintentando {retries}/{max_retries} en 5 segundos...")
                time.sleep(5)
            except Exception as e:
                retries += 1
                print(f"Error inesperado: {e}. Reintentando {retries}/{max_retries} en 5 segundos...")
                time.sleep(5)

        if not success:
            print(f"No se pudo descargar datos para rango {start_str} - {end_str}, saltando...")
            current_start = current_end
            continue

        if not klines:
            print(f"No hay datos para rango {start_str} - {end_str}, saltando...")
            current_start = current_end
            continue

        all_data.extend(klines)

        # Avanzar al siguiente chunk
        current_start = current_end
        time.sleep(0.3)

    if not all_data:
        print("No se descargaron datos.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def save_data():
    os.makedirs('data', exist_ok=True)
    
    timeframes = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE
    }

    for name, interval in timeframes.items():
        df = download_ohlcv_incremental(symbol, interval, lookback_days=30)
        df.to_csv(f'data/ohlcv_{name}.csv')
        print(f"Guardado: data/ohlcv_{name}.csv")

if __name__ == "__main__":
    save_data()
