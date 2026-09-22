# config.py
from __future__ import annotations

import os

# Credenciales de Binance Futures.
# Nunca se guardan valores reales en Git: configúralas en el entorno local.
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "").strip()
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "").strip()

SYMBOL = os.environ.get("BINANCE_SYMBOL", "BTCUSDT").strip() or "BTCUSDT"
USE_TESTNET = os.environ.get("BINANCE_USE_TESTNET", "true").strip().lower() not in {
    "0", "false", "no", "off"
}
