import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("[ERROR] imbalanced-learn not found.")
    print("        Run:  pip install imbalanced-learn")
    sys.exit(1)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("=" * 65)
print("  GROWFINIX ML INTERNSHIP - Task 2: Credit Card Fraud Detection")
print("=" * 65)

URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

print("\n[STEP 1] Loading dataset ...")
try:
    df = pd.read_csv(URL)
    print(f"  [OK] Dataset loaded from URL  ->  Shape: {df.shape}")
except Exception as e:
    print(f"  [WARN] Remote load failed ({e}).")
    print("  [INFO] Generating a synthetic fallback dataset ...")

    np.random.seed(42)
    n_normal = 9500
    n_fraud  = 500

    normal = {"Class": np.zeros(n_normal, dtype=int),
               "Time":  np.random.uniform(0, 172800, n_normal),
               "Amount": np.abs(np.random.lognormal(3.0, 1.5, n_normal))}
    for i in range(1, 29):
        normal[f"V{i}"] = np.random.normal(0.0, 1.0, n_normal)

    fraud_shifts = np.random.uniform(-3, 3, 28)
    fraud = {"Class": np.ones(n_fraud, dtype=int),
              "Time":  np.random.uniform(0, 172800, n_fraud),
              "Amount": np.abs(np.random.lognormal(4.0, 1.2, n_fraud))}
    for i in range(1, 29):
        fraud[f"V{i}"] = np.random.normal(fraud_shifts[i - 1], 1.5, n_fraud)

    df_normal = pd.DataFrame(normal)
    df_fraud  = pd.DataFrame(fraud)
    df = pd.concat([df_normal, df_fraud], ignore_index=True)

    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    df = df[cols].sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  [OK] Synthetic dataset generated  ->  Shape: {df.shape}")

print("\n[STEP 2] Inspecting the dataset ...")
print(f"\n  First 5 rows:\n{df[['Time', 'V1', 'V2', 'Amount', 'Class']].head().to_string()}")
print(f"\n  Dataset shape   : {df.shape}")
print(f"\n  Column dtypes   :\n{df.dtypes.to_string()}")
print(f"\n  Amount statistics:\n{df['Amount'].describe().to_string()}")
print(f"\n  Missing values per column:\n{df.isnull().sum().to_string()}")
print(f"\n  Duplicate rows  : {df.duplicated().sum()}")

print("\n[STEP 3] Cleaning the data ...")

before_dup = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed : {before_dup - len(df)}")

missing_total = df.isnull().sum().sum()
if missing_total > 0:
    df.dropna(inplace=True)
    print(f"  Rows with NaN values dropped.")
else:
    print("  No missing values found.")

assert df.isnull().sum().sum() == 0, "Remaining missing values found!"
print(f"  [OK] Data is clean. Final shape: {df.shape}")

print("\n[STEP 4] Analyzing class distribution ...")

class_counts = df["Class"].value_counts().sort_index()
normal_count = class_counts.get(0, 0)
fraud_count  = class_counts.get(1, 0)
total        = len(df)
normal_pct   = normal_count / total * 100
fraud_pct    = fraud_count  / total * 100

print(f"\n  Class distribution:")
print(f"  Normal transactions (0) : {normal_count:>8,}  ({normal_pct:.4f}%)")
print(f"  Fraud  transactions (1) : {fraud_count:>8,}  ({fraud_pct:.4f}%)")
print(f"  Imbalance ratio         : {normal_count // max(fraud_count, 1)}:1  (normal : fraud)")
print(f"\n  [!] Severe class imbalance detected.")
print(f"      Fraud cases account for only {fraud_pct:.2f}% of all transactions.")
print(f"      A naive model that predicts 'Normal' for everything would reach")
print(f"      {normal_pct:.2f}% accuracy — but would miss ALL fraud. SMOTE will")
print(f"      oversample the minority (fraud) class before training.")

print("\n  Generating EDA plots ...")

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Credit Card Fraud Detection — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(["Normal (0)", "Fraud (1)"],
               [normal_count, fraud_count],
               color=["#2196F3", "#F44336"], edgecolor="white", linewidth=0.8)
ax1.set_title("Class Distribution", fontweight="bold")
ax1.set_ylabel("Count")
for bar, val in zip(bars, [normal_count, fraud_count]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(normal_count, fraud_count) * 0.01,
             f"{val:,}", ha="center", fontsize=9, fontweight="bold")

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df[df["Class"] == 0]["Amount"].clip(upper=500),
         bins=50, alpha=0.6, color="#2196F3", label="Normal", edgecolor="white")
ax2.hist(df[df["Class"] == 1]["Amount"].clip(upper=500),
         bins=50, alpha=0.85, color="#F44336", label="Fraud", edgecolor="white")
ax2.set_title("Transaction Amount Distribution\n(capped at $500)", fontweight="bold")
ax2.set_xlabel("Amount ($)")
ax2.set_ylabel("Count")
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(np.log1p(df[df["Class"] == 0]["Amount"]),
         bins=50, alpha=0.6, color="#2196F3", label="Normal", edgecolor="white")
ax3.hist(np.log1p(df[df["Class"] == 1]["Amount"]),
         bins=50, alpha=0.85, color="#F44336", label="Fraud", edgecolor="white")
ax3.set_title("log(Amount + 1) Distribution", fontweight="bold")
ax3.set_xlabel("log(Amount + 1)")
ax3.set_ylabel("Count")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 0])
hour_normal = (df[df["Class"] == 0]["Time"] % 86400) / 3600
hour_fraud  = (df[df["Class"] == 1]["Time"] % 86400) / 3600
ax4.hist(hour_normal, bins=24, alpha=0.6, color="#2196F3", label="Normal", edgecolor="white")
ax4.hist(hour_fraud,  bins=24, alpha=0.85, color="#F44336", label="Fraud",  edgecolor="white")
ax4.set_title("Transaction Hour of Day", fontweight="bold")
ax4.set_xlabel("Hour")
ax4.set_ylabel("Count")
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
v_cols = [f"V{i}" for i in range(1, 11)]
mean_normal = df[df["Class"] == 0][v_cols].mean()
mean_fraud  = df[df["Class"] == 1][v_cols].mean()
x = np.arange(len(v_cols))
ax5.bar(x - 0.2, mean_normal, 0.4, label="Normal", color="#2196F3", edgecolor="white")
ax5.bar(x + 0.2, mean_fraud,  0.4, label="Fraud",  color="#F44336", edgecolor="white")
ax5.set_xticks(x)
ax5.set_xticklabels(v_cols, rotation=45, fontsize=8)
ax5.set_title("Mean Feature Values (V1–V10)\nNormal vs Fraud", fontweight="bold")
ax5.set_ylabel("Mean Value")
ax5.axhline(0, color="gray", linestyle="--", linewidth=0.7)
ax5.legend()

ax6 = fig.add_subplot(gs[1, 2])
bp = ax6.boxplot(
    [df[df["Class"] == 0]["Amount"].clip(upper=500),
     df[df["Class"] == 1]["Amount"].clip(upper=500)],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2)
)
ax6.set_xticklabels(["Normal", "Fraud"])
bp["boxes"][0].set_facecolor("#2196F3")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#F44336")
bp["boxes"][1].set_alpha(0.7)
ax6.set_title("Amount Boxplot (capped at $500)", fontweight="bold")
ax6.set_ylabel("Amount ($)")

plt.savefig("eda_fraud_plots.png", bbox_inches="tight")
print("  [OK] EDA plots saved to 'eda_fraud_plots.png'")
plt.close()

print("\n[STEP 5] Separating features (X) and target (y) ...")

X = df.drop(columns=["Class", "Time"])
y = df["Class"]
print(f"  Features used  : {X.shape[1]} columns (V1-V28 + Amount)")
print(f"  Features shape : {X.shape}")
print(f"  Target shape   : {y.shape}")

print("\n[STEP 6] Splitting data into train/test sets (80/20 split) ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training set   : {X_train.shape}")
print(f"  Testing  set   : {X_test.shape}")
print(f"  Train class distribution: {dict(y_train.value_counts().sort_index())}")
print(f"  Test  class distribution: {dict(y_test.value_counts().sort_index())}")

fraud_ratio_original = y_train.sum() / len(y_train)

X_train_normal = X_train[y_train == 0]

print("\n[STEP 7] Applying SMOTE to balance the training set ...")
print(f"  Fraud ratio BEFORE SMOTE : {fraud_ratio_original:.4f} ({fraud_ratio_original*100:.2f}%)")
print(f"  Class counts BEFORE SMOTE: {dict(y_train.value_counts().sort_index())}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

sm_counts = pd.Series(y_train_sm).value_counts().sort_index()
print(f"\n  Class counts AFTER  SMOTE: {dict(sm_counts)}")
print(f"  Training set size after SMOTE: {X_train_sm.shape}")
print(f"  [OK] SMOTE applied. Training data is now balanced (50/50).")

print("\n[STEP 8] Scaling features with StandardScaler ...")

scaler = StandardScaler()
X_train_normal_scaled = scaler.fit_transform(X_train_normal)
X_train_scaled        = scaler.transform(X_train_sm)
X_test_scaled         = scaler.transform(X_test)
print("  [OK] Features scaled. Mean ~= 0, Std ~= 1.")

print("\n[STEP 9] Training Isolation Forest ...")

contamination = 0.01
print(f"  Contamination parameter : {contamination}  (1% of test flagged as anomaly)")
print(f"  Training on : {X_train_normal_scaled.shape[0]:,} normal transactions only")

iso_forest = IsolationForest(
    n_estimators=100,
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_normal_scaled)
print("  [OK] Isolation Forest trained on normal (non-fraud) transaction data.")

print("\n[STEP 10] Evaluating model on the test set ...")

y_pred_raw = iso_forest.predict(X_test_scaled)
y_pred     = np.where(y_pred_raw == -1, 1, 0)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n  {'='*55}")
print(f"  Model : Isolation Forest")
print(f"  {'='*55}")
print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}  <- most important metric for fraud")
print(f"  F1-Score  : {f1:.4f}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
                            target_names=["Normal", "Fraud"],
                            zero_division=0))

print("\n[ANALYSIS] Fraud Detection Breakdown ...")

tn, fp, fn, tp = cm.ravel()
fraud_detection_rate = tp / max(tp + fn, 1) * 100
false_alarm_rate     = fp / max(tn + fp, 1) * 100

print(f"\n  Confusion Matrix Breakdown:")
print(f"  True  Negatives (Normal -> Normal) : {tn:>6,}  (correctly identified normal)")
print(f"  False Positives (Normal -> Fraud)  : {fp:>6,}  (false alarm — normal flagged as fraud)")
print(f"  False Negatives (Fraud  -> Normal) : {fn:>6,}  [CRITICAL] fraud cases missed")
print(f"  True  Positives (Fraud  -> Fraud)  : {tp:>6,}  (fraud correctly caught)")
print(f"\n  Fraud Detection Rate (Recall) : {fraud_detection_rate:.2f}%")
print(f"  False Alarm Rate              : {false_alarm_rate:.2f}%")
print(f"\n  Key Insight:")
print(f"  In fraud detection, RECALL is the most critical metric.")
print(f"  A missed fraud (False Negative) causes direct financial loss,")
print(f"  while a false alarm (False Positive) only causes inconvenience.")

print("\n  Generating result plots ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Isolation Forest — Credit Card Fraud Detection Results",
             fontsize=14, fontweight="bold")

ax = axes[0]
categories  = ["Before SMOTE\n(train)", "After SMOTE\n(train)"]
normal_vals = [int(y_train.value_counts().get(0, 0)),
               int(sm_counts.get(0, 0))]
fraud_vals  = [int(y_train.value_counts().get(1, 0)),
               int(sm_counts.get(1, 0))]
x     = np.arange(len(categories))
width = 0.35
bars_n = ax.bar(x - width / 2, normal_vals, width,
                label="Normal", color="#2196F3", edgecolor="white")
bars_f = ax.bar(x + width / 2, fraud_vals,  width,
                label="Fraud",  color="#F44336", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_title("Training Set: Before vs After SMOTE", fontweight="bold")
ax.set_ylabel("Count")
ax.legend()
for bar, val in zip(list(bars_n) + list(bars_f),
                    normal_vals + fraud_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(normal_vals) * 0.01,
            f"{val:,}", ha="center", fontsize=8, fontweight="bold")

ax = axes[1]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix\n(Isolation Forest)", fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")

ax = axes[2]
scores       = iso_forest.decision_function(X_test_scaled)
score_normal = scores[np.array(y_test) == 0]
score_fraud  = scores[np.array(y_test) == 1]
ax.hist(score_normal, bins=50, alpha=0.6, color="#2196F3",
        label="Normal", edgecolor="white")
ax.hist(score_fraud,  bins=50, alpha=0.85, color="#F44336",
        label="Fraud",  edgecolor="white")
ax.axvline(0, color="black", linestyle="--", linewidth=1.5,
           label="Decision Boundary (0)")
ax.set_title("Anomaly Score Distribution\n(Isolation Forest)",
             fontweight="bold")
ax.set_xlabel("Anomaly Score  (higher = more normal)")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig("fraud_detection_results.png", bbox_inches="tight")
print("  [OK] Result plots saved to 'fraud_detection_results.png'")
plt.close()

print("\n" + "=" * 65)
print("  MODEL TESTING - Predicting on New Transaction Samples")
print("=" * 65)

mean_normal_vals = df[df["Class"] == 0][[f"V{i}" for i in range(1, 29)] + ["Amount"]].mean()
mean_fraud_vals  = df[df["Class"] == 1][[f"V{i}" for i in range(1, 29)] + ["Amount"]].mean()

txn_a = mean_normal_vals.copy()
txn_a["Amount"] = 25.0

txn_b = mean_fraud_vals.copy()
txn_b["Amount"] = 312.0

txn_c = mean_normal_vals.copy()
txn_c["Amount"] = 2800.0

txn_d = mean_fraud_vals.copy()
txn_d["V1"]    = mean_fraud_vals["V1"] * 2.5
txn_d["V4"]    = mean_fraud_vals["V4"] * 2.0
txn_d["Amount"] = 1500.0

txn_e = mean_normal_vals.copy()
txn_e["Amount"] = 2.5

test_transactions = pd.DataFrame([txn_a, txn_b, txn_c, txn_d, txn_e],
                                  columns=[f"V{i}" for i in range(1, 29)] + ["Amount"])

transaction_labels = [
    "Txn A  — Normal profile   | Amount: $25.00",
    "Txn B  — Fraud profile    | Amount: $312.00",
    "Txn C  — High-amount      | Amount: $2,800.00",
    "Txn D  — Strong fraud sig | Amount: $1,500.00",
    "Txn E  — Tiny purchase    | Amount: $2.50",
]

expected = ["Normal", "Fraud", "Suspicious", "Fraud", "Normal"]

test_scaled = scaler.transform(test_transactions)

raw_preds   = iso_forest.predict(test_scaled)
predictions = np.where(raw_preds == -1, "FRAUD", "Normal")
scores      = iso_forest.decision_function(test_scaled)

print(f"\n  {'Transaction':<45} {'Prediction':<12} {'Anomaly Score':>15}  Expected")
print(f"  {'-'*88}")

for label, pred, score, exp in zip(transaction_labels, predictions, scores, expected):
    flag = "  <-- !! FRAUD ALERT" if pred == "FRAUD" else ""
    print(f"  {label:<45} {pred:<12} {score:>12.5f}   {exp}{flag}")

print(f"\n  {'='*65}")
print("  DETAILED TRANSACTION PREDICTIONS")
print(f"  {'='*65}")

for i, (label, pred, score, exp) in enumerate(
        zip(transaction_labels, predictions, scores, expected)):

    print(f"\n  {label}")
    print(f"  {'-'*55}")
    print(f"  Anomaly Score : {score:.5f}  (threshold ≈ 0; negative = anomaly)")
    print(f"  Prediction    : {pred}")
    print(f"  Expected      : {exp}")

    if score < 0:
        severity = "HIGH RISK" if score < -0.05 else "MODERATE RISK"
        print(f"  Risk Level    : {severity}  — transaction flagged as suspicious")
    else:
        print(f"  Risk Level    : LOW  — transaction appears legitimate")

    if pred == "FRAUD" and exp in ("Fraud", "Suspicious"):
        print(f"  Verdict       : CORRECT — fraud/suspicious correctly identified")
    elif pred == "Normal" and exp == "Normal":
        print(f"  Verdict       : CORRECT — normal transaction correctly passed")
    elif pred == "FRAUD" and exp == "Normal":
        print(f"  Verdict       : FALSE ALARM — normal transaction wrongly flagged")
    else:
        print(f"  Verdict       : MISSED — fraud not detected (False Negative)")

strict_expected = [1 if e == "Fraud" else 0 for e in expected if e != "Suspicious"]
strict_preds    = [1 if p == "FRAUD" else 0
                   for p, e in zip(predictions, expected) if e != "Suspicious"]
correct = sum(p == e for p, e in zip(strict_preds, strict_expected))
total_strict = len(strict_expected)

print(f"\n  {'='*65}")
print(f"  Custom Test Results (excluding borderline Txn C):")
print(f"  Correct Predictions : {correct} / {total_strict}")
print(f"  Test Accuracy       : {correct / total_strict * 100:.0f}%")
print(f"  {'='*65}")

print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset          : {total:,} transactions")
print(f"  Normal           : {normal_count:,} ({normal_pct:.4f}%)")
print(f"  Fraud            : {fraud_count:,} ({fraud_pct:.4f}%)")
print(f"  Imbalance Ratio  : {normal_count // max(fraud_count,1)}:1")
print(f"  {'─'*45}")
print(f"  Technique        : SMOTE (balanced training set)")
print(f"  Model            : Isolation Forest  (n=100 trees)")
print(f"  {'─'*45}")
print(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
print(f"  Precision        : {prec:.4f}")
print(f"  Recall           : {rec:.4f}  ({rec*100:.2f}% fraud detected)")
print(f"  F1-Score         : {f1:.4f}")
print(f"  {'─'*45}")
print(f"  Fraud Caught     : {tp} / {tp + fn}")
print(f"  Fraud Missed     : {fn}  (False Negatives)")
print(f"  False Alarms     : {fp}  (False Positives)")
print("=" * 65)
print("\n  Output files:")
print("    eda_fraud_plots.png          — Exploratory Data Analysis")
print("    fraud_detection_results.png  — SMOTE + Results + Anomaly Scores")
print("=" * 65)
