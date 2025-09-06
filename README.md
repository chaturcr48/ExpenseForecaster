<<<<<<< HEAD
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
=======
# Introduction 
TODO: Give a short introduction of your project. Let this section explain the objectives or the motivation behind this project. 

# Getting Started
TODO: Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# Build and Test
TODO: Describe and show how to build your code and run the tests. 

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)
>>>>>>> 088fb2a85afe6a7b68e1811519e2234c102b851d
