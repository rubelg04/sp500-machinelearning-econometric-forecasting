# S&P 500 Forecasting

Code accompanying my dissertation on S&P 500 return forecasting.

## Files

- `sp500_random_walk.py` — Random walk benchmark
- `sp500_arima513_forecast.py` — ARIMA(5,1,3) walk-forward forecast
- `sp500_svm_forecast.py` — Support Vector Regression forecast
- `sp500_transformer_forecast.py` — Transformer ensemble forecast
- `sp500_significance_tests.py` — Pesaran-Timmermann and Diebold-Mariano tests

## Data

All scripts use yfinance to download S&P 500 (^GSPC) data. No API key required.
