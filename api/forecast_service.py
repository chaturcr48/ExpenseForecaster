# api/forecast_service.py
import pandas as pd
from fastapi import FastAPI, Query
from pathlib import Path
from pydantic import BaseModel
from retrain_forecast import retrain_and_save

app = FastAPI(title="Expense Forecaster Agent")

# Load forecast data prepared in Step 3
data_path = Path("artifacts/category_forecasts_next12.csv")
df_forecasts = pd.read_csv(data_path, parse_dates=["month"])
df_monthly_expenses = pd.read_csv(Path("data/expenses_monthly.csv"), parse_dates=["month"])  

@app.get("/")
def home():
    return {"message": "Welcome to the Expense Forecaster Agent 🚀"}

@app.post("/retrain")
def retrain():
    """Retrain models on latest data and update forecasts."""
    global df_forecasts
    df_forecasts = retrain_and_save()
    return {"message": "✅ Models retrained and forecasts updated"}

@app.get("/monthly-expenses-data")
def get_monthly_expenses_data():
    """Get all monthly expenses data."""
    return df_monthly_expenses.to_dict(orient="records")

@app.get("/all-category-forecast-data")
def get_all_category_forecast_data():
    """Get all category forecast data."""
    return df_forecasts.to_dict(orient="records")

@app.get("/forecast")
def get_forecast(
    category: str = Query("Payroll", description="Expense category (e.g. Travel, Payroll, ALL)"),
    months: int = Query(3, ge=1, le=12, description="Number of months to forecast (1–12)")
):
    # Validate category
    if category not in df_forecasts["category"].unique():
        return {"error": f"Invalid category. Choose from {df_forecasts['category'].unique().tolist()}"}
    
    # Get forecast subset
    result = (df_forecasts[df_forecasts["category"] == category]
                .sort_values("month")
                .head(months))
    
    # Convert datetime to str for JSON
    output = [
        {"month": row.month.strftime("%Y-%m"), "forecast_amount": float(row.forecast_amount)}
        for row in result.itertuples(index=False)
    ]
    
    return {
        "category": category,
        "months_requested": months,
        "forecast": output
    }


# @app.get("/forecast")
# def get_forecast(category: str, months: int = 6):
#     df_cat = df_forecasts[df_forecasts["category"] == category].head(months)

#     result = {
#         "category": category,
#         "months_requested": months,
#         "forecast": df_cat[["month", "forecast_amount", "lower_ci", "upper_ci"]]
#                     .to_dict(orient="records")
#     }
#     return result



# class ForecastRequest(BaseModel):
#     category: str
#     months: int

# @app.post("/forecast")
# def get_forecast(request: ForecastRequest):
#     df_cat = df_forecasts[df_forecasts["category"] == request.category].head(request.months)

#     return {
#         "category": request.category,
#         "months_requested": request.months,
#         "forecast": df_cat[["month", "forecast_amount", "lower_ci", "upper_ci"]]
#                     .to_dict(orient="records")
#     }

