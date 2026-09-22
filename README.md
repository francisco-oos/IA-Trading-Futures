# IA-Trading-Futures

Proyecto de trading algorítmico usando inteligencia artificial (XGBoost) para operar en Binance Futures automáticamente.

## Características

- Descarga de datos OHLCV desde Binance
- Entrenamiento de modelo XGBoost
- Predicción de señales (compra / venta)
- Backtesting básico
- Preparado para operar con Binance Futures vía API

## Requisitos

- Python 3.10+
- Cuenta en Binance Testnet con claves API
- Paquetes: pandas, numpy, xgboost, matplotlib, ta, python-binance

## Estructura


## Configuración segura

Las credenciales no deben escribirse en `config.py` ni subirse al repositorio. Defínelas sólo en el equipo donde se ejecute el proyecto.

```powershell
$env:BINANCE_API_KEY = "TU_API_KEY"
$env:BINANCE_API_SECRET = "TU_API_SECRET"
$env:BINANCE_USE_TESTNET = "true"
python main.py
```

En Linux/macOS use `export BINANCE_API_KEY=...` y `export BINANCE_API_SECRET=...`.

El archivo `.env.example` documenta los nombres esperados, pero este proyecto lee directamente variables del entorno y no requiere guardar un `.env`.
