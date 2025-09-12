import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# ---------------- SARIMAX ----------------
def forecast_sarimax(series, cat):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    future_fc = res.get_forecast(steps=12)
    f_mean = future_fc.predicted_mean.reset_index()
    f_ci = future_fc.conf_int()

    f_mean.columns = ["month", "forecast_amount"]
    f_mean["lower_ci"] = f_ci.iloc[:,0].values
    f_mean["upper_ci"] = f_ci.iloc[:,1].values
    f_mean["category"] = cat
    f_mean["month"] = pd.to_datetime(f_mean["month"]).dt.to_period("M").dt.to_timestamp("M")
    return f_mean

# ---------------- Holt-Winters ----------------
def forecast_holtwinters(series, cat):
    model = ExponentialSmoothing(series, seasonal="add", seasonal_periods=12)
    res = model.fit()

    f_mean = res.forecast(12).reset_index()
    f_mean.columns = ["month", "forecast_amount"]
    f_mean["lower_ci"] = f_mean["forecast_amount"] * 0.9  # simple 10% band
    f_mean["upper_ci"] = f_mean["forecast_amount"] * 1.1
    f_mean["category"] = cat
    f_mean["month"] = pd.to_datetime(f_mean["month"]).dt.to_period("M").dt.to_timestamp("M")
    return f_mean

# ---------------- Prophet ----------------
def forecast_prophet(series, cat):
    df_prophet = series.reset_index()
    df_prophet.columns = ["ds", "y"]

    model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    f_mean = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12).rename(
        columns={"ds":"month", "yhat":"forecast_amount", "yhat_lower":"lower_ci", "yhat_upper":"upper_ci"}
    )
    f_mean["category"] = cat
    f_mean["month"] = pd.to_datetime(f_mean["month"]).dt.to_period("M").dt.to_timestamp("M")
    return f_mean

from pathlib import Path

def retrain_and_save():
    """Retrain SARIMAX, HoltWinters, Prophet and save ensemble forecasts."""
    data_path = Path("data/expenses_monthly.csv")
    df = pd.read_csv(data_path, parse_dates=["month"])

    categories = [c for c in df["category"].unique() if c != "ALL"]
    all_forecasts = []

    for cat in categories:
        series = df[df["category"] == cat].set_index("month").sort_index()["amount"]

        forecasts = []
        for forecaster in [forecast_sarimax, forecast_holtwinters, forecast_prophet]:
            forecasts.append(forecaster(series, cat))

        # Merge by month
        df_merged = forecasts[0][["month"]].copy()
        for f in forecasts:
            df_merged = df_merged.merge(f[["month","forecast_amount"]], on="month", how="left")

        # Average forecast values
        df_merged["forecast_amount"] = df_merged.iloc[:,1:].mean(axis=1)
        df_merged["lower_ci"] = df_merged.iloc[:,1:].min(axis=1)
        df_merged["upper_ci"] = df_merged.iloc[:,1:].max(axis=1)
        df_merged["category"] = cat

        all_forecasts.append(df_merged[["month","forecast_amount","lower_ci","upper_ci","category"]])

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

    print(f"✅ Ensemble forecasts retrained and saved: {out_csv.resolve()}")
    return df_final
