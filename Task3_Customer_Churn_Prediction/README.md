# Task 3: E-Commerce – Customer Churn Prediction

**Growfinix Machine Learning Internship | Task 3 | Domain: E-Commerce**

---

## Objective

Predict which customers are likely to **stop engaging** with an e-commerce platform (churn) based on their past activity, purchase history, and engagement metrics. Compare **Random Forest** and **XGBoost** classifiers and identify the better-performing model.

---

## Dataset

A synthetic e-commerce customer dataset is generated directly by the script — no external download required. It contains **5,000 customer records** with realistic feature distributions.

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Customer age (18–70) |
| `Gender` | Categorical | Male / Female |
| `TenureMonths` | Numeric | Months as a registered customer (1–60) |
| `TotalPurchases` | Numeric | Total number of orders placed |
| `AvgOrderValue` | Numeric | Average order value in USD |
| `DaysSinceLastPurchase` | Numeric | Days since most recent order (recency) |
| `NumComplaints` | Numeric | Number of support complaints raised |
| `EmailOpenRate` | Numeric | Percentage of marketing emails opened (0–100%) |
| `LoyaltyPoints` | Numeric | Accumulated loyalty program points |
| `ReturnsRate` | Numeric | Percentage of orders returned (0–50%) |
| `SubscriptionTier` | Categorical | Free / Basic / Premium |
| `PreferredCategory` | Categorical | Electronics / Clothing / Food / Home / Sports |
| `PaymentMethod` | Categorical | Credit Card / Debit Card / UPI / Net Banking |
| **`Churn`** | **Target** | **1 = Churned, 0 = Retained** |

### Churn Signal Design

Churn probability is modelled using real-world e-commerce logic:

| Factor | Direction |
|---|---|
| Inactive > 60 days | ↑ higher churn risk |
| Free subscription tier | ↑ higher churn risk |
| Low email open rate (<30%) | ↑ higher churn risk |
| ≥2 complaints | ↑ higher churn risk |
| Tenure < 6 months | ↑ higher churn risk |
| Few purchases (<20) | ↑ higher churn risk |

---

## Steps Performed

| Step | Description |
|---|---|
| 1 | Generate a 5,000-row synthetic e-commerce dataset |
| 2 | Inspect shape, dtypes, statistics, missing values |
| 3 | Drop CustomerID, handle duplicates and missing values |
| 4 | Perform EDA — churn distribution, tenure, recency, tier analysis |
| 5 | Separate features (X) and target (y) |
| 6 | 80/20 stratified train/test split |
| 7 | Label-encode categorical columns; StandardScaler on numeric columns |
| 8 | Train **Random Forest** (200 trees) and **XGBoost** (200 trees) |
| 9 | Evaluate both models: accuracy, precision, recall, F1, confusion matrix |
| 10 | Compare models and identify the **winner by F1-score** |
| 11 | Show top-10 highest-risk customers and predict 5 custom profiles |

---

## Models

### Random Forest
- An ensemble of 200 decision trees
- Each tree votes; majority wins
- Naturally handles non-linear relationships and feature interactions
- Robust to outliers

### XGBoost (Extreme Gradient Boosting)
- Builds trees sequentially; each corrects previous errors
- Highly effective on tabular data
- Regularization via `subsample` and `colsample_bytree`

---

## Preprocessing

| Step | Method | Detail |
|---|---|---|
| Categorical encoding | `LabelEncoder` | Fit on training data only — no leakage |
| Numeric scaling | `StandardScaler` | Fit on training data only — no leakage |
| Columns encoded | Gender, SubscriptionTier, PreferredCategory, PaymentMethod | |

---

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Run

```bash
python customer_churn_prediction.py
```

> **No dataset download required.** The script generates the dataset automatically.

---

## Model Testing — Custom Customer Profiles

After training, the script predicts churn for **5 hand-crafted customer profiles**:

| Customer | Profile | Expected |
|---|---|---|
| Cust A | 48-month tenure, Premium, 85 purchases, active (5 days) | No Churn |
| Cust B | 2-month tenure, Free, 3 purchases, inactive (150 days), 4 complaints | Churn |
| Cust C | 24-month tenure, Basic, 40 purchases, 45 days since last order | Moderate |
| Cust D | 1-month tenure, Free, 1 purchase, 200 days inactive, 5 complaints | Churn |
| Cust E | 56-month tenure, Premium, 130 purchases, active (10 days) | No Churn |

For each customer the script prints:
- **Random Forest prediction + churn probability**
- **XGBoost prediction + churn probability**
- Whether models agree or disagree

---

## Output Files

| File | Description |
|---|---|
| `eda_churn_plots.png` | 6-panel EDA: churn distribution, tenure, recency, tier, email rate, order value |
| `model_comparison_churn.png` | Side-by-side metrics bar chart + both confusion matrices |
| `feature_importance_churn.png` | Top-12 important features for RF and XGBoost |

---

## Key Design Decisions

- **Synthetic dataset** — purpose-built with e-commerce feature names and realistic distributions; no external dependency
- **LabelEncoder** (not OneHotEncoder) — both RF and XGBoost are tree-based and handle ordinal-encoded categoricals correctly
- **Scalers/encoders fit on training only** — strict prevention of data leakage
- **Stratified split** — preserves churn/no-churn ratio in both train and test sets
- **F1-score as winner criterion** — better than accuracy on imbalanced classes; balances precision and recall

---

## Project Structure

```
Task3_Customer_Churn_Prediction/
├── customer_churn_prediction.py   ← Main Python script
└── README.md                      ← This file
```
