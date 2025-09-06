# retrain_forecast.py
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

def retrain_and_save():
    """Retrain SARIMAX for each category and save forecasts."""
    data_path = Path("data/expenses_monthly.csv")
    df = pd.read_csv(data_path, parse_dates=["month"])

    categories = [c for c in df["category"].unique() if c != "ALL"]
    all_forecasts = []

    for cat in categories:
        series = df[df["category"] == cat].set_index("month").sort_index()["amount"]

        # Train SARIMAX
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)

        # Forecast next 12 months with conf intervals
        future_fc = res.get_forecast(steps=12)
        f_mean = future_fc.predicted_mean.reset_index()
        f_ci = future_fc.conf_int()

        f_mean.columns = ["month", "forecast_amount"]
        f_mean["category"] = cat
        f_mean["lower_ci"] = f_ci.iloc[:,0].values
        f_mean["upper_ci"] = f_ci.iloc[:,1].values
        all_forecasts.append(f_mean)

    # Combine forecasts
    df_fc = pd.concat(all_forecasts, ignore_index=True)

    # Add total = sum across categories
    df_total = (
        df_fc.groupby("month")
        .agg(
            forecast_amount=("forecast_amount", "sum"),
            lower_ci=("lower_ci", "sum"),
            upper_ci=("upper_ci", "sum")
        )
        .reset_index())
    
    df_total["category"] = "ALL"
    df_final = pd.concat([df_fc, df_total], ignore_index=True)

    out_csv = Path("artifacts/category_forecasts_next12.csv")
    df_final.to_csv(out_csv, index=False)

    print(f"✅ Forecasts retrained and saved: {out_csv.resolve()}")
    return df_final
