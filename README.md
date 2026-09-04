# 🚀 Growfinix – Machine Learning Internship Projects

A comprehensive collection of machine learning, data science, and time-series forecasting projects developed during the **Growfinix Machine Learning Internship**. This repository features end-to-end solutions across healthcare, finance, e-commerce, retail, and energy domains.

---

## 📌 Repository Overview

| # | Task | Domain | Problem Type | Core Algorithms / Techniques | Key Results / Metric |
|---|---|---|---|---|---|
| **01** | [Heart Disease Prediction](./Task1_Heart_Disease_Prediction) | **Healthcare** | Binary Classification | Logistic Regression, SVM (RBF kernel), StandardScaler | High accuracy with robust recall for disease identification |
| **02** | [Credit Card Fraud Detection](./Task2_Credit_Card_Fraud_Detection) | **Finance** | Imbalanced Classification | SMOTE, Random Forest, Logistic Regression, Anomaly Scoring | Effectively captured rare fraud cases without precision collapse |
| **03** | [Customer Churn Prediction](./Task3_Customer_Churn_Prediction) | **E-Commerce** | Binary Classification | Random Forest, XGBoost, Feature Importance Analysis | High ROC-AUC; identified key behavioural drivers of churn |
| **04** | [Market Basket Analysis](./Task4_Market_Basket_Analysis) | **Retail** | Association Rule Mining | Apriori Algorithm, Support, Confidence, Lift, Heatmap Analysis | Actionable multi-item cross-sell and co-purchase recommendations |
| **05** | [Power Consumption Forecasting](./Task5_Power_Consumption_Forecasting) | **Energy** | Time Series Forecasting | ARIMA, ADF Stationarity Test, Seasonality Decomposition, ACF/PACF | Accurate demand forecasting with low MAE and RMSE |

---

## 📂 Project Structure

```
Growfinix/
├── Task1_Heart_Disease_Prediction/
│   ├── heart_disease_prediction.py       # Full training & evaluation pipeline
│   ├── eda_plots.png                     # Exploratory data analysis visualizations
│   ├── model_comparison.png              # Comparison chart (LR vs SVM)
│   └── README.md                         # Detailed documentation for Task 1
│
├── Task2_Credit_Card_Fraud_Detection/
│   ├── credit_card_fraud_detection.py    # Pipeline with SMOTE balancing
│   ├── eda_fraud_plots.png               # Fraud distribution & transaction analysis
│   ├── fraud_detection_results.png       # SMOTE impact & confusion matrices
│   └── README.md                         # Detailed documentation for Task 2
│
├── Task3_Customer_Churn_Prediction/
│   ├── customer_churn_prediction.py      # RF & XGBoost comparison pipeline
│   ├── eda_churn_plots.png               # Churn correlation & demographic insights
│   ├── feature_importance_churn.png      # Feature importance rankings
│   ├── model_comparison_churn.png        # Evaluation curves & metric comparison
│   └── README.md                         # Detailed documentation for Task 3
│
├── Task4_Market_Basket_Analysis/
│   ├── market_basket_analysis.py         # Apriori frequent itemsets & rules mining
│   ├── itemset_heatmap.png               # Association rule confidence & lift heatmap
│   ├── market_basket_analysis.png        # Transaction distribution & network plots
│   └── README.md                         # Detailed documentation for Task 4
│
├── Task5_Power_Consumption_Forecasting/
│   ├── power_consumption_forecasting.py # 5-year power consumption ARIMA pipeline
│   ├── acf_pacf_plot.png                 # Autocorrelation and partial autocorrelation
│   ├── eda_power_plots.png               # Trend, seasonality, and residual breakdown
│   ├── forecast_power.png                # Multi-step ahead forecast with intervals
│   └── README.md                         # Detailed documentation for Task 5
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack & Dependencies

- **Programming Language**: Python 3.8+
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Gradient Boosting**: `xgboost`
- **Imbalanced Learning**: `imbalanced-learn` (SMOTE)
- **Association Rules**: `mlxtend`
- **Time Series & Econometrics**: `statsmodels`

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityachanguli37-a11y/Growfinix.git
   cd Growfinix
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Running the Projects

Each task is self-contained with synthetic or benchmark dataset generators and automated plot exports:

```bash
# Task 1: Heart Disease Prediction
python Task1_Heart_Disease_Prediction/heart_disease_prediction.py

# Task 2: Credit Card Fraud Detection
python Task2_Credit_Card_Fraud_Detection/credit_card_fraud_detection.py

# Task 3: Customer Churn Prediction
python Task3_Customer_Churn_Prediction/customer_churn_prediction.py

# Task 4: Market Basket Analysis
python Task4_Market_Basket_Analysis/market_basket_analysis.py

# Task 5: Power Consumption Forecasting
python Task5_Power_Consumption_Forecasting/power_consumption_forecasting.py
```

---

## 📜 Task Highlights & Summaries

### 🩺 Task 1: Healthcare – Heart Disease Prediction
- Evaluated cardiovascular indicators (blood pressure, cholesterol, resting ECG, fluoroscopy results).
- Trained Logistic Regression and Support Vector Machine (SVM) classifiers with standard feature scaling.
- Generated comprehensive confusion matrices and ROC curves.

### 💳 Task 2: Finance – Credit Card Fraud Detection
- Tackled extreme class imbalance typical in financial transaction monitoring.
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on training splits to prevent data leakage.
- Evaluated Precision-Recall AUC and cost-sensitive classification thresholds.

### 👥 Task 3: E-Commerce – Customer Churn Prediction
- Modeled subscriber behavior using transaction recency, frequency, tenure, and customer support tickets.
- Benchmarked Random Forest against XGBoost, uncovering top churn risk factors through feature importance metrics.

### 🛒 Task 4: Retail – Market Basket Analysis
- Implemented the **Apriori Algorithm** to discover frequent itemsets and derive actionable association rules.
- Analyzed cross-sell potential using Support, Confidence, and Lift metrics visualized in pairwise heatmaps.

### ⚡ Task 5: Energy – Power Consumption Forecasting
- Analyzed 5 years of electricity demand patterns, accounting for monthly trends and seasonal peaks.
- Validated stationarity with the Augmented Dickey-Fuller (ADF) test, identified AR and MA orders via ACF/PACF plots, and forecasted future energy demand using an ARIMA model.

---

## 👤 Author

Developed by **[Aditya C S](https://github.com/adityachanguli37-a11y)** as part of the **Growfinix Internship**.
