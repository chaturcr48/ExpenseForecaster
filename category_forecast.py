# step3_category_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Load data ---
data_path = Path("data/expenses_monthly.csv")
df = pd.read_csv(data_path, parse_dates=["month"])

categories = [c for c in df["category"].unique() if c != "ALL"]

# Store all forecasts
all_forecasts = []

for cat in categories:
    series = df[df["category"] == cat].set_index("month").sort_index()["amount"]

    # Train/Test split (last 6 months for test)
    train = series.iloc[:-6]
    test = series.iloc[-6:]

    # Fit SARIMAX (simple seasonal)
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # Forecast future 12 months
    future_fc = res.get_forecast(steps=12)
    f_mean = future_fc.predicted_mean.reset_index()
    f_mean.columns = ["month", "forecast_amount"]
    f_mean["category"] = cat
    all_forecasts.append(f_mean)

    # --- Plot each category ---
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test (actual)", color="black")
    plt.plot(f_mean["month"], f_mean["forecast_amount"], label="Forecast", color="red")
    plt.title(f"Forecast for {cat}")
    plt.legend()
    plt.tight_layout()
    out_img = Path(f"artifacts/forecast_{cat}.png")
    plt.savefig(out_img)
    plt.close()
    print(f"✅ Saved: {out_img}")

# --- Combine all forecasts ---
df_fc = pd.concat(all_forecasts, ignore_index=True)

# Add total = sum across categories
df_total = (df_fc.groupby("month")["forecast_amount"].sum()
             .reset_index())
df_total["category"] = "ALL"

df_final = pd.concat([df_fc, df_total], ignore_index=True)

# Save to CSV
out_csv = Path("artifacts/category_forecasts_next12.csv")
df_final.to_csv(out_csv, index=False)

print(f"\n✅ Combined forecast saved to: {out_csv.resolve()}")
print(df_final.head(10))
