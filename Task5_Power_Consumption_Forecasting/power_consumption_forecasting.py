import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
    print("[ERROR] statsmodels not found.")
    print("        Run: pip install statsmodels")
    sys.exit(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("=" * 65)
print("  GROWFINIX ML INTERNSHIP - Task 5: Power Consumption Forecasting")
print("=" * 65)

print("\n[STEP 1] Generating 5-year monthly power consumption dataset ...")

np.random.seed(42)
N = 60
dates = pd.date_range(start="2019-01-01", periods=N, freq="MS")

trend = np.linspace(0, 800, N)

seasonal_pattern = np.array([
    380,
    340,
    120,
    -40,
     20,
    180,
    420,
    390,
     80,
    -60,
     10,
    220,
])
seasonal = np.tile(seasonal_pattern, 5)

noise = np.random.normal(0, 45, N)

consumption = 5200 + trend + seasonal + noise

df = pd.DataFrame({
    "Date":                 dates,
    "PowerConsumption_MWh": np.round(consumption, 2),
})

print(f"  [OK] Dataset generated  -> {df.shape[0]} monthly records")
print(f"  Period : {df['Date'].min().strftime('%b %Y')}  to  {df['Date'].max().strftime('%b %Y')}")

print("\n[STEP 2] Inspecting the dataset ...")
print(f"\n  First 6 rows:\n{df.head(6).to_string(index=False)}")
print(f"\n  Shape        : {df.shape}")
print(f"  Dtypes       :\n{df.dtypes.to_string()}")
print(f"\n  Statistics:\n{df['PowerConsumption_MWh'].describe().round(2).to_string()}")
print(f"\n  Missing values : {df.isnull().sum().sum()}")
print(f"  Duplicate rows : {df.duplicated().sum()}")

print("\n[STEP 3] Cleaning the data ...")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed : {before - len(df)}")

missing = df["PowerConsumption_MWh"].isnull().sum()
if missing > 0:
    df["PowerConsumption_MWh"] = df["PowerConsumption_MWh"].interpolate(method="linear")
    print(f"  Missing values interpolated : {missing}")
else:
    print("  No missing values found.")

assert df.isnull().sum().sum() == 0
print(f"  [OK] Data is clean. Shape: {df.shape}")

print("\n[STEP 4] Converting to time-series format ...")

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df.index.freq = "MS"

series = df["PowerConsumption_MWh"].copy()

print(f"  DatetimeIndex set. Frequency : {series.index.freqstr}")
print(f"  Index range  : {series.index[0].strftime('%b %Y')} to {series.index[-1].strftime('%b %Y')}")
print(f"  Total months : {len(series)}")

print("\n[STEP 5] Analyzing electricity consumption trends ...")

rolling_mean = series.rolling(window=12).mean()
rolling_std  = series.rolling(window=12).std()
yearly       = series.resample("YE").sum()

print(f"\n  Overall mean   : {series.mean():>10.2f} MWh")
print(f"  Overall std    : {series.std():>10.2f} MWh")
print(f"  Min            : {series.min():>10.2f} MWh  ({series.idxmin().strftime('%b %Y')})")
print(f"  Max            : {series.max():>10.2f} MWh  ({series.idxmax().strftime('%b %Y')})")
print(f"\n  Annual total consumption:")
for yr, val in yearly.items():
    print(f"    {yr.year} : {val:>12,.2f} MWh")

print("\n[STEP 6] Analyzing seasonality ...")

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
monthly_avg = series.groupby(series.index.month).mean()

print("\n  Average power consumption by month (all 5 years):")
mn_min = monthly_avg.min()
mn_max = monthly_avg.max()
for mn, val in zip(month_names, monthly_avg.values):
    bar_len = int((val - mn_min) / (mn_max - mn_min) * 30)
    print(f"  {mn:>3}: {val:>8.1f} MWh  {'█' * bar_len}")

decomp = seasonal_decompose(series, model="additive", period=12)
print(f"\n  Seasonal decomposition:")
print(f"  Trend range   : {decomp.trend.dropna().min():.1f}  to  {decomp.trend.dropna().max():.1f} MWh")
print(f"  Seasonal range: {decomp.seasonal.min():.1f}  to  {decomp.seasonal.max():.1f} MWh")
print(f"  Residual std  : {decomp.resid.dropna().std():.1f} MWh")

print("\n  Generating EDA plots ...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Power Consumption Forecasting — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.32)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(series.index, series.values,
         color="#2196F3", linewidth=1.6, label="Monthly Consumption")
ax1.plot(rolling_mean.index, rolling_mean.values,
         color="#F44336", linewidth=2.2, linestyle="--", label="12-Month Rolling Mean")
ax1.fill_between(series.index,
                 rolling_mean - rolling_std, rolling_mean + rolling_std,
                 alpha=0.12, color="#F44336", label="±1 Std Dev Band")
ax1.set_title("Monthly Power Consumption (2019–2023) with Rolling Statistics",
              fontweight="bold")
ax1.set_xlabel("Date")
ax1.set_ylabel("Consumption (MWh)")
ax1.legend()

ax2 = fig.add_subplot(gs[1, 0])
monthly_data = [series[series.index.month == m].values for m in range(1, 13)]
bp = ax2.boxplot(monthly_data, patch_artist=True,
                 medianprops=dict(color="white", linewidth=2))
cmap = plt.cm.coolwarm
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(cmap(i / 11))
    patch.set_alpha(0.8)
ax2.set_xticklabels(month_names, fontsize=8)
ax2.set_title("Monthly Seasonality Pattern (Boxplot)", fontweight="bold")
ax2.set_ylabel("Consumption (MWh)")

ax3 = fig.add_subplot(gs[1, 1])
yr_years = [yr.year for yr in yearly.index]
yr_vals  = yearly.values
bars = ax3.bar(yr_years, yr_vals, color="#2196F3", edgecolor="white")
ax3.set_title("Annual Total Power Consumption", fontweight="bold")
ax3.set_ylabel("Total Consumption (MWh)")
ax3.set_xlabel("Year")
for bar, val in zip(bars, yr_vals):
    ax3.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 100, f"{val:,.0f}",
             ha="center", fontsize=8, fontweight="bold")

ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(decomp.trend.dropna().index, decomp.trend.dropna().values,
         color="#4CAF50", linewidth=2)
ax4.set_title("Trend Component (Additive Decomposition)", fontweight="bold")
ax4.set_ylabel("Trend (MWh)")
ax4.set_xlabel("Date")

ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(decomp.seasonal.index[:24], decomp.seasonal.values[:24],
         color="#FF9800", linewidth=2, label="First 2 years shown")
ax5.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax5.set_title("Seasonal Component (Additive Decomposition)", fontweight="bold")
ax5.set_ylabel("Seasonal Effect (MWh)")
ax5.set_xlabel("Date")
ax5.legend(fontsize=8)

plt.savefig("eda_power_plots.png", bbox_inches="tight")
print("  [OK] EDA plots saved to 'eda_power_plots.png'")
plt.close()

print("\n[STEP 7] Preparing data for time-series forecasting ...")

train = series.iloc[:-12]
test  = series.iloc[-12:]
forecast_date = test.index[-1] + pd.DateOffset(months=1)

print(f"  Training set  : {train.index[0].strftime('%b %Y')} – {train.index[-1].strftime('%b %Y')}  ({len(train)} months)")
print(f"  Test set      : {test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')}   ({len(test)} months)")
print(f"  Forecast date : {forecast_date.strftime('%B %Y')}")

print("\n  Stationarity Test (Augmented Dickey-Fuller):")
adf_stat, adf_p, _, _, adf_crit, _ = adfuller(train, autolag="AIC")
print(f"  ADF Statistic  : {adf_stat:.4f}")
print(f"  p-value        : {adf_p:.4f}")
for k, v in adf_crit.items():
    print(f"  Critical {k}   : {v:.4f}")

if adf_p > 0.05:
    d = 1
    print("  [INFO] Series is NON-STATIONARY (p > 0.05) -> will use d=1")
else:
    d = 0
    print("  [INFO] Series is STATIONARY (p <= 0.05) -> will use d=0")

print("\n  Selecting ARIMA order via AIC grid search (p, q in {0,1,2}) ...")
candidates  = [(p, d, q) for p in range(3) for q in range(3)]
aic_results = []

for order in candidates:
    try:
        res = ARIMA(train, order=order).fit()
        aic_results.append((order, round(res.aic, 2)))
    except Exception:
        pass

if not aic_results:
    best_order = (1, d, 1)
    print(f"  [WARN] Grid search failed. Using default ARIMA(1,{d},1).")
else:
    aic_results.sort(key=lambda x: x[1])
    best_order = aic_results[0][0]
    print(f"\n  Top 5 ARIMA orders by AIC:")
    print(f"  {'Order':<16} {'AIC':>10}")
    print(f"  {'-'*28}")
    for i, (o, a) in enumerate(aic_results[:5]):
        tag = "  <- SELECTED" if i == 0 else ""
        print(f"  ARIMA{str(o):<10}  {a:>10.2f}{tag}")

print(f"\n  Selected : ARIMA{best_order}")

print("\n  Generating ACF / PACF plots ...")
acf_series = train.diff().dropna() if d >= 1 else train

fig_acf, axes_acf = plt.subplots(1, 2, figsize=(14, 4))
fig_acf.suptitle(
    f"ACF and PACF — {'Differenced (d=1)' if d == 1 else 'Original'} Training Series",
    fontweight="bold"
)
plot_acf(acf_series,  lags=20, ax=axes_acf[0])
plot_pacf(acf_series, lags=20, ax=axes_acf[1], method="ywm")
axes_acf[0].set_title("Autocorrelation Function (ACF)", fontweight="bold")
axes_acf[1].set_title("Partial Autocorrelation Function (PACF)", fontweight="bold")
plt.tight_layout()
plt.savefig("acf_pacf_plot.png", bbox_inches="tight")
print("  [OK] ACF/PACF plot saved to 'acf_pacf_plot.png'")
plt.close()

print(f"\n[STEP 8] Training ARIMA{best_order} model ...")

arima_model = ARIMA(train, order=best_order)
model_fit   = arima_model.fit()

print("  [OK] ARIMA model fitted.")
print(f"\n  Model summary (key stats):")
print(f"  AIC            : {model_fit.aic:.2f}")
print(f"  BIC            : {model_fit.bic:.2f}")
print(f"  Log-Likelihood : {model_fit.llf:.2f}")

n_steps = len(test) + 1
print(f"\n[STEP 9] Forecasting {n_steps} steps ahead (test + upcoming month) ...")

fc_obj  = model_fit.get_forecast(steps=n_steps)
fc_mean = fc_obj.predicted_mean
fc_ci   = fc_obj.conf_int(alpha=0.05)

test_fc_vals  = fc_mean.values[:len(test)]
test_fc       = pd.Series(test_fc_vals, index=test.index)
ci_lower_test = fc_ci.values[:len(test), 0]
ci_upper_test = fc_ci.values[:len(test), 1]

next_fc    = float(fc_mean.values[-1])
next_lower = float(fc_ci.values[-1, 0])
next_upper = float(fc_ci.values[-1, 1])

print(f"\n  Upcoming Month Power Demand Forecast")
print(f"  Month          : {forecast_date.strftime('%B %Y')}")
print(f"  {'─'*42}")
print(f"  Predicted      : {next_fc:>12,.2f} MWh")
print(f"  95% CI (lower) : {next_lower:>12,.2f} MWh")
print(f"  95% CI (upper) : {next_upper:>12,.2f} MWh")
print(f"  Margin (±)     : {(next_upper - next_lower) / 2:>12,.2f} MWh")

print("\n[STEP 10] Evaluating forecast on the test period ...")

mae  = mean_absolute_error(test.values, test_fc.values)
rmse = float(np.sqrt(mean_squared_error(test.values, test_fc.values)))
mape = float(np.mean(np.abs((test.values - test_fc.values) / test.values)) * 100)

print(f"\n  Test period: {test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')}")
print(f"  {'─'*52}")
print(f"  MAE   (Mean Absolute Error)      : {mae:>10.2f} MWh")
print(f"  RMSE  (Root Mean Squared Error)  : {rmse:>10.2f} MWh")
print(f"  MAPE  (Mean Abs Percentage Error): {mape:>10.2f} %")
print(f"  Model Accuracy  (100 - MAPE)     : {100 - mape:>10.2f} %")

print(f"\n  Actual vs Forecast breakdown:")
print(f"  {'Month':<10} {'Actual (MWh)':>14} {'Forecast':>14} {'Error':>10} {'Abs %':>8}")
print(f"  {'-'*60}")
for dt, act, fct in zip(test.index, test.values, test_fc.values):
    err  = act - fct
    absp = abs(err) / act * 100
    print(f"  {dt.strftime('%b %Y'):<10} {act:>14.2f} {fct:>14.2f} {err:>+10.2f} {absp:>7.2f}%")

print("\n[STEP 11] Generating forecast visualisation ...")

fig, axes = plt.subplots(2, 1, figsize=(16, 11))
fig.suptitle(f"Power Consumption Forecast — ARIMA{best_order}",
             fontsize=15, fontweight="bold")

ax = axes[0]
ax.plot(train.index, train.values,
        color="#2196F3", linewidth=1.5,
        label=f"Training Data ({train.index[0].strftime('%b %Y')} – {train.index[-1].strftime('%b %Y')})")
ax.plot(test.index, test.values,
        color="#4CAF50", linewidth=2, marker="o", markersize=4,
        label=f"Actual (Test: {test.index[0].strftime('%b %Y')} – {test.index[-1].strftime('%b %Y')})")
ax.plot(test.index, test_fc.values,
        color="#F44336", linewidth=2, linestyle="--", marker="s", markersize=4,
        label=f"ARIMA{best_order} Forecast")
ax.fill_between(test.index, ci_lower_test, ci_upper_test,
                alpha=0.18, color="#F44336", label="95% Confidence Interval")
ax.scatter([forecast_date], [next_fc],
           color="#9C27B0", s=150, zorder=6,
           label=f"Upcoming Forecast  {forecast_date.strftime('%b %Y')} = {next_fc:,.0f} MWh")
ax.errorbar([forecast_date], [next_fc],
            yerr=[[next_fc - next_lower], [next_upper - next_fc]],
            fmt="none", color="#9C27B0", capsize=7, linewidth=2)
ax.axvline(test.index[0], color="gray", linestyle=":", linewidth=1.5, label="Train / Test Split")
ax.set_title("Full Series: Training + Test Forecast + Upcoming Month Projection",
             fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Power Consumption (MWh)")
ax.legend(fontsize=8, loc="upper left")

ax2 = axes[1]
zoom_start  = series.index[-20]
zoom_train  = train[train.index >= zoom_start]
ax2.plot(zoom_train.index, zoom_train.values,
         color="#2196F3", linewidth=2, label="Training Data")
ax2.plot(test.index, test.values,
         color="#4CAF50", linewidth=2.2, marker="o", markersize=6,
         label="Actual (2023)")
ax2.plot(test.index, test_fc.values,
         color="#F44336", linewidth=2.2, linestyle="--", marker="s", markersize=6,
         label=f"ARIMA{best_order} Forecast")
ax2.fill_between(test.index, ci_lower_test, ci_upper_test,
                 alpha=0.22, color="#F44336", label="95% CI")
ax2.scatter([forecast_date], [next_fc],
            color="#9C27B0", s=180, zorder=6,
            label=f"{forecast_date.strftime('%b %Y')} Forecast: {next_fc:,.0f} MWh  ← UPCOMING")
ax2.errorbar([forecast_date], [next_fc],
             yerr=[[next_fc - next_lower], [next_upper - next_fc]],
             fmt="none", color="#9C27B0", capsize=9, linewidth=2.2)
ax2.axvline(test.index[0], color="gray", linestyle=":", linewidth=1.5, label="Train / Test Split")
ax2.set_title(f"Zoomed View: Last 20 Months + {forecast_date.strftime('%B %Y')} Forecast",
              fontweight="bold")
ax2.set_xlabel("Date")
ax2.set_ylabel("Power Consumption (MWh)")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("forecast_power.png", bbox_inches="tight")
print("  [OK] Forecast plot saved to 'forecast_power.png'")
plt.close()

print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset          : 60 monthly records (Jan 2019 – Dec 2023)")
print(f"  Training period  : Jan 2019 – Dec 2022  (48 months)")
print(f"  Test period      : Jan 2023 – Dec 2023  (12 months)")
print(f"  Model            : ARIMA{best_order}")
print(f"  {'─'*45}")
print(f"  MAE              : {mae:>10.2f} MWh")
print(f"  RMSE             : {rmse:>10.2f} MWh")
print(f"  MAPE             : {mape:>10.2f} %")
print(f"  Accuracy         : {100 - mape:>10.2f} %")
print(f"  {'─'*45}")
print(f"  Upcoming Forecast: {forecast_date.strftime('%B %Y')}")
print(f"  Predicted Demand : {next_fc:>10,.2f} MWh")
print(f"  95% CI           : [{next_lower:,.2f}  –  {next_upper:,.2f}] MWh")
print("=" * 65)
print("\n  Output files:")
print("    eda_power_plots.png   — Trends, seasonality, decomposition")
print("    acf_pacf_plot.png     — ACF and PACF of training series")
print("    forecast_power.png    — Actual vs forecast + upcoming month")
print("=" * 65)
