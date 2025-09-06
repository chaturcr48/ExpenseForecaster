# step2_train_forecast.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --- Load data ---
data_path = Path("data/expenses_monthly.csv")
df = pd.read_csv(data_path, parse_dates=["month"])
df_all = df[df["category"] == "ALL"].set_index("month").sort_index()

# Use "amount" series
y = df_all["amount"]

# --- Train/Test split ---
train = y.iloc[:-6]   # train on all but last 6 months
test  = y.iloc[-6:]   # holdout

# --- Fit SARIMAX ---
# (p,d,q) × (P,D,Q, s=12 months seasonality)
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
res = model.fit(disp=False)

print(res.summary())

# --- Forecast next 6 months (matching test) ---
forecast = res.get_forecast(steps=6)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()

# --- Plot actual vs forecast ---
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test (actual)", color="black")
plt.plot(pred_mean.index, pred_mean, label="Forecast", color="red")
plt.fill_between(pred_ci.index, pred_ci.iloc[:,0], pred_ci.iloc[:,1],
                 color="pink", alpha=0.3)
plt.title("Expense Forecast (Baseline SARIMAX)")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/sarimax_forecast.png")
plt.show()

# --- Forecast next 12 months (future prediction) ---
future_forecast = res.get_forecast(steps=12)
f_mean = future_forecast.predicted_mean.reset_index()
f_mean.columns = ["month","forecast_amount"]

out_path = Path("artifacts/forecast_next12.csv")
f_mean.to_csv(out_path, index=False)

print(f"\n✅ Saved forecast plot: artifacts/sarimax_forecast.png")
print(f"✅ Saved 12-month forecast CSV: {out_path.resolve()}")
print(f_mean.head())
