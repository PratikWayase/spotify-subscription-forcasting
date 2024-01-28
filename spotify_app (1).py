from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

app = FastAPI()

# Load data
df = pd.read_csv("spotify_Subscriptions.csv")
df["Time Period"] = pd.to_datetime(df["Time Period"], format='%d/%m/%Y')

# Calculate quarterly growth
df["quarterly growth"] = df["Subscribers"].pct_change() * 100

# Calculate yearly growth
df["year"] = df["Time Period"].dt.year
df["yearly_growth"] = df.groupby('year')["Subscribers"].pct_change() * 100

# Using ARIMA for Forecasting spotify Subscriptions
time_series = df.set_index('Time Period')['Subscribers']
differenced_series = time_series.diff().dropna()

# ARIMA model
p, d, q = 1, 1, 1
model = ARIMA(time_series, order=(p, d, q))
results = model.fit()

# Forecast future steps
future_steps = 7
predictions = results.predict(len(time_series), len(time_series) + future_steps - 1)
predictions = predictions.astype(int)

@app.get("/")
def read_root():
    return {"message": "Welcome to Spotify Subscriptions Analysis and Forecasting API"}

@app.get("/original_data")
def get_original_data():
    return df.to_dict(orient='records')

@app.get("/quarterly_growth_plot")
def get_quarterly_growth_plot():
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Time Period', y='quarterly growth', data=df)
    plt.title('Spotify Subscriptions Growth Rate')
    plt.xlabel('Time Period')
    plt.ylabel('Quarterly Growth Rate (%)')
    plt.xticks(rotation=50, ha='right')
    return plt

@app.get("/yearly_growth_plot")
def get_yearly_growth_plot():
    plt.figure(figsize=(10, 6))
    sns.barplot(x="year", y="yearly_growth", data=df)
    plt.title('Spotify Yearly Subscriptions Growth Rate')
    plt.xlabel('Time Period')
    plt.ylabel('Quarterly Growth Rate (%)')
    plt.xticks(rotation=45, ha='right')
    return plt

@app.get("/arima_forecast")
def get_arima_forecast():
    return {"summary": results.summary().tables[1].as_html(), "predictions": predictions.to_dict()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
