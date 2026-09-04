# Task 2: Finance – Credit Card Fraud Detection (Imbalanced Data)
### Growfinix Machine Learning Internship

---

## 📌 Objective

Detect fraudulent credit card transactions from a **heavily imbalanced** dataset where fraud cases represent a tiny fraction of all transactions.

The pipeline uses:
- **SMOTE** (Synthetic Minority Over-sampling Technique) from `imbalanced-learn` to balance the training set
- **Isolation Forest** from `scikit-learn` for unsupervised anomaly-based fraud detection

---

## 📁 Project Structure

```
Task2_Credit_Card_Fraud_Detection/
│
├── credit_card_fraud_detection.py   # Main Python script (all steps)
├── eda_fraud_plots.png              # EDA visualisations (auto-generated)
├── fraud_detection_results.png      # SMOTE + results + anomaly scores (auto-generated)
└── README.md                        # This file
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) / TensorFlow public mirror |
| **Rows** | 284,807 transactions (2 days of European cardholder data) |
| **Features** | 30 (Time, V1–V28 PCA-anonymised, Amount) |
| **Target** | Binary — `Class`: 0 = Normal, 1 = Fraud |
| **Fallback** | Synthetic dataset generated automatically if URL is unavailable |

> **Note**: V1–V28 are the result of PCA applied by the dataset authors to protect cardholder privacy. The original feature names are not disclosed.

### Feature Descriptions

| Feature | Description |
|---|---|
| `Time` | Seconds elapsed since the first transaction (dropped before training) |
| `V1`–`V28` | PCA-transformed anonymised features |
| `Amount` | Transaction amount in EUR |
| `Class` | Target label — 0 = Normal, 1 = Fraud |

---

## ⚠️ Class Imbalance

The dataset is **severely imbalanced**:

| Class | Count | Percentage |
|---|---|---|
| Normal (0) | 284,315 | ~99.83% |
| Fraud  (1) | 492 | ~0.17% |

A naive model predicting "Normal" for everything would achieve 99.83% accuracy — but would **miss every single fraud case**. This makes accuracy a misleading metric here; **Recall** is the most important metric.

---

## 🚀 Steps Performed

| Step | Description |
|---|---|
| 1 | Load dataset from URL (with synthetic fallback) |
| 2 | Inspect shape, dtypes, Amount statistics, missing values |
| 3 | Remove duplicates and handle missing values |
| 4 | Analyze class distribution and visualize imbalance |
| 5 | Separate features (X) and target (y) — drop `Time` column |
| 6 | 80/20 stratified train/test split |
| 7 | Apply **SMOTE** to the training set only (balance to 50/50) — shown for analysis |
| 8 | Scale features with `StandardScaler` (fit on normal-only training data) |
| 9 | Train **Isolation Forest** on **normal transactions only** (learns "what normal looks like") |
| 10 | Evaluate on original (unbalanced) test set — Isolation Forest flags deviations as fraud |
| Analysis | Detailed breakdown: TP, TN, FP, FN, fraud detection rate, false alarm rate |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

Requires Python 3.8+

---

## How to Run

```bash
# Navigate to the project folder
cd Task2_Credit_Card_Fraud_Detection

# Run the script
python credit_card_fraud_detection.py
```

The script will:
1. Try to download the creditcard dataset from the TensorFlow public mirror.
2. If the download fails, automatically generate a realistic synthetic fallback dataset.
3. Run all 10 steps and print a complete report to the terminal.
4. Save two plot images to the current directory.

---

## Model Testing — Custom Transaction Samples

After training and evaluation, the script tests the model on **5 hand-crafted transactions** to demonstrate real-world usage:

| Transaction | Profile | Amount | Expected |
|---|---|---|---|
| Txn A | Typical normal feature values | $25.00 | Normal |
| Txn B | Typical fraud feature values | $312.00 | Fraud |
| Txn C | Normal features, unusually high amount | $2,800.00 | Suspicious |
| Txn D | Strong fraud signal (amplified features) | $1,500.00 | Fraud |
| Txn E | Tiny low-risk purchase, normal profile | $2.50 | Normal |

For each transaction the script prints:
- **Anomaly Score** — the raw Isolation Forest score (negative = anomaly)
- **Prediction** — `Normal` or `FRAUD`
- **Risk Level** — `LOW`, `MODERATE RISK`, or `HIGH RISK`
- **Verdict** — whether the prediction was correct, a false alarm, or a miss

> The V1–V28 values are derived from the **mean feature values per class** computed directly from the loaded dataset, ensuring the test samples are statistically grounded.

---

## Output Files

| File | Description |
|---|---|
| `eda_fraud_plots.png` | Class distribution, amount histograms, log-amount, hourly pattern, V1–V10 feature means, amount boxplot |
| `fraud_detection_results.png` | SMOTE before/after chart, confusion matrix, Isolation Forest anomaly score distribution |

---

## Evaluation Metrics

The model is evaluated on the **original (unbalanced) test set** using:

- **Accuracy** — Overall correct predictions (misleading on imbalanced data alone)
- **Precision** — Of all transactions flagged as fraud, how many are actually fraud?
- **Recall** — Of all actual fraud cases, how many were caught? *(most critical)*
- **F1-Score** — Harmonic mean of Precision and Recall
- **Confusion Matrix** — TP, TN, FP, FN breakdown
- **Classification Report** — Per-class precision, recall, F1, support

> **Why Recall matters most**: A missed fraud (False Negative) causes direct financial loss. A false alarm (False Positive) causes minor inconvenience. The model is optimised to minimise missed fraud.

---

## How Isolation Forest Works

Isolation Forest detects anomalies by randomly partitioning data using decision trees. The key insight:

- **Normal points** require many splits to isolate (they cluster together).
- **Anomalies (fraud)** are isolated with far fewer splits (they are sparse/different).

**Training strategy**: The model is trained exclusively on **normal transactions**. It learns the statistical structure of what a legitimate transaction looks like. When applied to the test set, transactions that don't fit that structure are flagged as fraud.

The `contamination` parameter (set to `0.01`) controls the decision threshold — it tells the model what fraction of test samples to classify as outliers. A value of `0.01` (1%) provides a good balance of Recall and Precision on this dataset.

**Output mapping:**
| Isolation Forest Output | Meaning | Mapped Label |
|---|---|---|
| `1` | Inlier (normal) | `0` (Normal) |
| `-1` | Outlier (anomaly) | `1` (Fraud) |

---

## How SMOTE Works

SMOTE creates **synthetic** minority-class samples by interpolating between existing fraud examples and their nearest neighbours — rather than simply duplicating them. This prevents overfitting on repeated examples and produces a more generalisable training set.

Applied **only to the training set** to avoid information leakage into the test set.

---

## Key Design Decisions

- **`Time` column dropped** — it is a sequential counter with no fraud-pattern signal
- **Isolation Forest trained on normal-only data** — the standard anomaly detection paradigm; the model learns normal behaviour and flags deviations
- **SMOTE applied for analysis** — illustrates class balancing; the balanced distribution is shown in the results plot
- **contamination = 0.01** — a practical threshold; the true rate (0.17%) is too small for reliable IF boundary calibration in PCA space
- **Stratified split** — maintains the fraud/normal ratio across train and test sets
- **StandardScaler fit on normal training data** — ensures `Amount` is normalised without leaking any test statistics

---

## About

**Internship**: Growfinix Machine Learning Internship
**Task**: Task 2 – Finance – Credit Card Fraud Detection (Imbalanced Data)
**Domain**: Finance / Anomaly Detection / Binary Classification
**Tools**: Python, Pandas, NumPy, Scikit-learn (Isolation Forest), imbalanced-learn (SMOTE), Matplotlib, Seaborn
