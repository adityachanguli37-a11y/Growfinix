# Task 5: Energy – Power Consumption Forecasting (Time Series)

**Growfinix Machine Learning Internship | Task 5 | Domain: Energy**

---

## Objective

Forecast the **upcoming month's electricity demand** using historical monthly power consumption data spanning **5 years**. The model uses **ARIMA (AutoRegressive Integrated Moving Average)** from Statsmodels to capture trend and autocorrelation patterns in the time series.

---

## Dataset

A synthetic monthly electricity consumption dataset is generated directly by the script — no external download required. It contains **60 monthly records** (January 2019 – December 2023).

| Column | Type | Description |
|---|---|---|
| `Date` | DatetimeIndex | Month-start timestamp (MS frequency) |
| `PowerConsumption_MWh` | Float | Monthly electricity consumed (MWh) |

### Dataset Design (Realistic Components)

| Component | Description |
|---|---|
| **Base load** | 5,200 MWh/month |
| **Trend** | Slow linear growth of +800 MWh over 5 years (rising demand) |
| **Seasonality** | Peaks in Jan-Feb (winter) and Jul-Aug (summer); troughs in Apr, Oct |
| **Noise** | Gaussian noise (σ = 45 MWh) for realistic variability |

---

## Steps Performed

| Step | Description |
|---|---|
| 1 | Generate a 60-month synthetic power consumption dataset |
| 2 | Inspect shape, dtypes, descriptive statistics, missing values |
| 3 | Remove duplicates; interpolate any missing values |
| 4 | Set DatetimeIndex with monthly start (`MS`) frequency |
| 5 | Analyze trends — rolling 12-month mean/std, annual totals |
| 6 | Analyze seasonality — monthly boxplot, additive decomposition |
| 7 | Prepare for ARIMA — train/test split, ADF stationarity test, ACF/PACF, AIC grid search |
| 8 | Train the selected **ARIMA(p,d,q)** model on the training set |
| 9 | Forecast the **upcoming month** (January 2024) with 95% confidence intervals |
| 10 | Evaluate on the 12-month test set using **MAE, RMSE, MAPE** |
| 11 | Plot actual vs forecasted power consumption (full view + zoomed view) |

---

## Model: ARIMA

**ARIMA(p, d, q)** = AutoRegressive Integrated Moving Average

| Parameter | Meaning |
|---|---|
| **p** | Number of AR (autoregressive) terms — past values |
| **d** | Degree of differencing — makes series stationary |
| **q** | Number of MA (moving average) terms — past forecast errors |

### How the order is selected

1. **ADF Test** — checks if the series is stationary; if `p-value > 0.05`, set `d=1`.
2. **AIC Grid Search** — tries all combinations of `p ∈ {0,1,2}`, `q ∈ {0,1,2}` with the selected `d`. The order with the **lowest AIC** is chosen.
3. **ACF/PACF plots** — saved as `acf_pacf_plot.png` for manual inspection.

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **MAE** | `mean(|actual - forecast|)` | Average absolute error in MWh |
| **RMSE** | `sqrt(mean((actual - forecast)²))` | Penalises large errors more |
| **MAPE** | `mean(|error / actual|) × 100` | Intuitive percentage accuracy |

---

## Train / Test Split

| Split | Period | Months |
|---|---|---|
| **Training** | Jan 2019 – Dec 2022 | 48 months |
| **Test** | Jan 2023 – Dec 2023 | 12 months |
| **Forecast** | Jan 2024 | 1 month |

---

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib statsmodels scikit-learn
```

### Run

```bash
python power_consumption_forecasting.py
```

> **No dataset download required.** The script generates data automatically.

---

## Output Files

| File | Description |
|---|---|
| `eda_power_plots.png` | Rolling statistics, monthly seasonality boxplot, annual totals, trend & seasonal decomposition |
| `acf_pacf_plot.png` | ACF and PACF of the (differenced) training series |
| `forecast_power.png` | Full view and zoomed view: actual vs forecast, 95% CI, upcoming month highlighted |

---

## Key Design Decisions

- **Synthetic dataset** — purpose-built with realistic trend, dual seasonality (winter + summer peaks), and Gaussian noise; self-contained, no API dependency
- **ADF stationarity test** — objectively determines the differencing order `d`
- **AIC grid search** — selects `p` and `q` by minimising information criterion, not by manual guessing
- **48-month training / 12-month test split** — provides sufficient test examples to compute meaningful MAE/RMSE/MAPE
- **95% confidence intervals** — shown on both the test forecast and the upcoming month prediction, communicating forecast uncertainty clearly
- **get_forecast()** — uses the statsmodels `get_forecast()` API (preferred over `forecast()`) to obtain both point estimates and full confidence intervals

---

## Project Structure

```
Task5_Power_Consumption_Forecasting/
├── power_consumption_forecasting.py   ← Main Python script
└── README.md                          ← This file
```
