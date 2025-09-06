# Expense Forecaster

This project forecasts expenses for different categories using SARIMAX time series models. It is designed for PepsiCo Food Canada and provides retraining and forecasting scripts, as well as an API for serving forecasts.

## Project Structure

- `bootstrap.py` — Project setup or initialization script.
- `category_forecast.py` — Category-level forecasting logic.
- `retrain_forecast.py` — Retrains SARIMAX models for each category and saves forecasts.
- `train_forecast.py` — Initial training of forecasting models.
- `api/forecast_service.py` — API service for serving forecasts.
- `artifacts/` — Output files, including forecast CSVs and images.
- `data/expenses_monthly.csv` — Source data for expense forecasting.
- `models/` — Saved model files (if any).
- `scripts/` — Utility or helper scripts.

## How to Use

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Retrain forecasts:**
   ```
   python retrain_forecast.py
   ```

   This will read `data/expenses_monthly.csv`, retrain SARIMAX models for each category, and save the next 12 months' forecasts to `artifacts/category_forecasts_next12.csv`.

3. **API Service:**
   See `api/forecast_service.py` for details on running the API.

## Requirements

- Python 3.8+
- See `requirements.txt` for required packages.

## Output

- Forecast CSVs and images are saved in the `artifacts/` directory.

## License

[Add your license here]