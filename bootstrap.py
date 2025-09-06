# step1_bootstrap.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

# --- Folders ---
for d in ["data", "artifacts", "models", "scripts", "api"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# --- Config ---
np.random.seed(42)
start, end = "2021-01-01", "2025-08-01"   # 56 months of history
months = pd.date_range(start, end, freq="MS")  # month start

categories = {
    "Payroll":   {"base": 120_000, "trend": 0.0010, "noise": 0.02},
    "Cloud":     {"base":  45_000, "trend": 0.0030, "noise": 0.07},
    "Travel":    {"base":  18_000, "trend": 0.0020, "noise": 0.15},
    "Marketing": {"base":  25_000, "trend": 0.0015, "noise": 0.12},
    "Misc":      {"base":   8_000, "trend": 0.0005, "noise": 0.25},
}

# Monthly seasonal multipliers (1..12)
# Heavier spend around Mar/Apr (events), Jun (cloud), Oct/Nov (festive), Dec (year-end)
seasonality = np.array([0.98, 1.00, 1.06, 1.08, 1.03, 1.05, 1.00, 0.97, 0.99, 1.07, 1.10, 1.12])

def make_series(base, trend, noise):
    t = np.arange(len(months))
    m = months.month - 1  # 0..11
    seas = seasonality[m]
    # multiplicative trend and seasonality with log-normal noise
    raw = base * (1 + trend * t) * seas
    eps = np.random.normal(0, noise, size=len(months))
    vals = raw * np.exp(eps)  # positive, heteroscedastic-ish
    return np.maximum(vals, 0.0)

rows = []
for cat, cfg in categories.items():
    y = make_series(**cfg)
    rows.extend([{"month": m, "category": cat, "amount": float(a)} for m, a in zip(months, y)])

# Add ALL (sum of categories)
df = pd.DataFrame(rows)
tot = (df.pivot_table(index="month", columns="category", values="amount", aggfunc="sum")
         .sum(axis=1)
         .rename("amount")
         .reset_index())
tot["category"] = "ALL"
df_all = pd.concat([df, tot[["month","category","amount"]]], ignore_index=True)

# Save aggregated monthly data
out_path = Path("data/expenses_monthly.csv")
df_all.sort_values(["category","month"]).to_csv(out_path, index=False)

# Small printout
print(f"✅ Generated {len(months)} months for {len(categories)} categories (+ ALL).")
print(f"Saved: {out_path.resolve()}")
print(df_all.head(8))
