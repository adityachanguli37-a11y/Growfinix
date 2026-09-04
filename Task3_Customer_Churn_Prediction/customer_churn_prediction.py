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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("[ERROR] xgboost not found.")
    print("        Run: pip install xgboost")
    sys.exit(1)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# STEP 1 - Load / Generate the Dataset
# =============================================================================
print("=" * 65)
print("  GROWFINIX ML INTERNSHIP - Task 3: Customer Churn Prediction")
print("=" * 65)
print("\n[STEP 1] Generating synthetic e-commerce customer dataset ...")

np.random.seed(42)
n = 5000

tenure      = np.random.randint(1, 61, n)
age         = np.random.randint(18, 71, n)
gender      = np.random.choice(["Male", "Female"], n)

total_purchases     = np.clip(np.random.poisson(30, n), 0, 200)
avg_order_value     = np.round(np.clip(np.random.lognormal(4.2, 0.8, n), 10, 800), 2)
days_since_last     = np.clip(np.random.exponential(45, n).astype(int), 0, 365)
num_complaints      = np.clip(np.random.poisson(1, n), 0, 20)
email_open_rate     = np.round(np.random.beta(3, 4, n) * 100, 1)
loyalty_points      = np.random.randint(0, 10001, n)
returns_rate        = np.round(np.random.beta(1.5, 6, n) * 50, 1)
subscription_tier   = np.random.choice(["Free", "Basic", "Premium"], n, p=[0.35, 0.40, 0.25])
preferred_category  = np.random.choice(
    ["Electronics", "Clothing", "Food", "Home", "Sports"], n
)
payment_method      = np.random.choice(
    ["Credit Card", "Debit Card", "UPI", "Net Banking"], n
)

is_inactive     = (days_since_last > 60).astype(float)
is_new          = (tenure < 6).astype(float)
has_complaints  = (num_complaints >= 2).astype(float)
low_engagement  = (email_open_rate < 30).astype(float)
is_free         = (subscription_tier == "Free").astype(float)
few_purchases   = (total_purchases < 20).astype(float)

p_churn = (
    0.30 * is_inactive +
    0.25 * is_free +
    0.20 * low_engagement +
    0.15 * has_complaints +
    0.15 * is_new +
    0.10 * few_purchases +
    0.05 * np.random.random(n)
)
p_churn = np.clip(p_churn, 0, 0.95)
churn   = (np.random.random(n) < p_churn).astype(int)

df = pd.DataFrame({
    "CustomerID":            [f"EC-{i:05d}" for i in range(n)],
    "Age":                   age,
    "Gender":                gender,
    "TenureMonths":          tenure,
    "TotalPurchases":        total_purchases,
    "AvgOrderValue":         avg_order_value,
    "DaysSinceLastPurchase": days_since_last,
    "NumComplaints":         num_complaints,
    "EmailOpenRate":         email_open_rate,
    "LoyaltyPoints":         loyalty_points,
    "ReturnsRate":           returns_rate,
    "SubscriptionTier":      subscription_tier,
    "PreferredCategory":     preferred_category,
    "PaymentMethod":         payment_method,
    "Churn":                 churn,
})

print(f"  [OK] Dataset generated  ->  Shape: {df.shape}")

# =============================================================================
# STEP 2 - Inspect the Dataset
# =============================================================================
print("\n[STEP 2] Inspecting the dataset ...")
print(f"\n  First 5 rows:")
print(df[["CustomerID", "TenureMonths", "TotalPurchases",
          "AvgOrderValue", "SubscriptionTier", "Churn"]].head().to_string())
print(f"\n  Dataset shape   : {df.shape}")
print(f"\n  Column dtypes   :\n{df.dtypes.to_string()}")
print(f"\n  Numeric statistics:")
print(df[["Age", "TenureMonths", "TotalPurchases", "AvgOrderValue",
          "DaysSinceLastPurchase", "NumComplaints",
          "EmailOpenRate", "LoyaltyPoints"]].describe().round(2).to_string())
print(f"\n  Missing values per column:\n{df.isnull().sum().to_string()}")
print(f"\n  Duplicate rows  : {df.duplicated().sum()}")

# =============================================================================
# STEP 3 - Clean the Data
# =============================================================================
print("\n[STEP 3] Cleaning the data ...")

df.drop(columns=["CustomerID"], inplace=True)
print("  CustomerID dropped (not a predictive feature).")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed : {before - len(df)}")

missing = df.isnull().sum().sum()
if missing > 0:
    df.dropna(inplace=True)
    print(f"  Rows with NaN dropped. Remaining: {len(df)}")
else:
    print("  No missing values found.")

assert df.isnull().sum().sum() == 0
print(f"  [OK] Data is clean. Final shape: {df.shape}")

# =============================================================================
# STEP 4 - Basic Data Analysis (EDA)
# =============================================================================
print("\n[STEP 4] Basic data analysis ...")

churn_counts = df["Churn"].value_counts().sort_index()
churn_pct    = df["Churn"].value_counts(normalize=True).sort_index() * 100

print(f"\n  Churn distribution:")
print(f"  No Churn (0) : {churn_counts.get(0,0):>5,}  ({churn_pct.get(0,0):.2f}%)")
print(f"  Churned  (1) : {churn_counts.get(1,0):>5,}  ({churn_pct.get(1,0):.2f}%)")
print(f"\n  Mean TenureMonths by Churn:\n{df.groupby('Churn')['TenureMonths'].mean().round(2).to_string()}")
print(f"\n  Mean DaysSinceLastPurchase by Churn:\n{df.groupby('Churn')['DaysSinceLastPurchase'].mean().round(2).to_string()}")
print(f"\n  Mean EmailOpenRate by Churn:\n{df.groupby('Churn')['EmailOpenRate'].mean().round(2).to_string()}")
print(f"\n  SubscriptionTier vs Churn:")
print(pd.crosstab(df["SubscriptionTier"], df["Churn"],
                  colnames=["Churn"], rownames=["Tier"]).to_string())

print("\n  Generating EDA plots ...")

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Customer Churn Prediction — Exploratory Data Analysis",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(["No Churn", "Churned"],
               [churn_counts.get(0, 0), churn_counts.get(1, 0)],
               color=["#2196F3", "#F44336"], edgecolor="white")
ax1.set_title("Churn Distribution", fontweight="bold")
ax1.set_ylabel("Number of Customers")
for bar, val in zip(bars, [churn_counts.get(0, 0), churn_counts.get(1, 0)]):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 15, f"{val:,}", ha="center", fontweight="bold")

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(df[df["Churn"] == 0]["TenureMonths"], bins=25, alpha=0.6,
         color="#2196F3", label="No Churn", edgecolor="white")
ax2.hist(df[df["Churn"] == 1]["TenureMonths"], bins=25, alpha=0.85,
         color="#F44336", label="Churned", edgecolor="white")
ax2.set_title("Tenure Distribution by Churn", fontweight="bold")
ax2.set_xlabel("Tenure (months)")
ax2.set_ylabel("Count")
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(df[df["Churn"] == 0]["DaysSinceLastPurchase"], bins=25, alpha=0.6,
         color="#2196F3", label="No Churn", edgecolor="white")
ax3.hist(df[df["Churn"] == 1]["DaysSinceLastPurchase"], bins=25, alpha=0.85,
         color="#F44336", label="Churned", edgecolor="white")
ax3.set_title("Days Since Last Purchase", fontweight="bold")
ax3.set_xlabel("Days")
ax3.set_ylabel("Count")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 0])
tier_churn = df.groupby("SubscriptionTier")["Churn"].value_counts().unstack(fill_value=0)
tier_churn.columns = ["No Churn", "Churned"]
tier_churn = tier_churn.reindex(["Free", "Basic", "Premium"])
tier_churn.plot(kind="bar", ax=ax4, color=["#2196F3", "#F44336"],
                edgecolor="white", rot=0)
ax4.set_title("Subscription Tier vs Churn", fontweight="bold")
ax4.set_ylabel("Count")
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(df[df["Churn"] == 0]["EmailOpenRate"], bins=25, alpha=0.6,
         color="#2196F3", label="No Churn", edgecolor="white")
ax5.hist(df[df["Churn"] == 1]["EmailOpenRate"], bins=25, alpha=0.85,
         color="#F44336", label="Churned", edgecolor="white")
ax5.set_title("Email Open Rate by Churn", fontweight="bold")
ax5.set_xlabel("Email Open Rate (%)")
ax5.set_ylabel("Count")
ax5.legend()

ax6 = fig.add_subplot(gs[1, 2])
bp = ax6.boxplot(
    [df[df["Churn"] == 0]["AvgOrderValue"],
     df[df["Churn"] == 1]["AvgOrderValue"]],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2)
)
ax6.set_xticklabels(["No Churn", "Churned"])
bp["boxes"][0].set_facecolor("#2196F3")
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#F44336")
bp["boxes"][1].set_alpha(0.7)
ax6.set_title("Avg Order Value by Churn", fontweight="bold")
ax6.set_ylabel("Avg Order Value ($)")

plt.savefig("eda_churn_plots.png", bbox_inches="tight")
print("  [OK] EDA plots saved to 'eda_churn_plots.png'")
plt.close()

# =============================================================================
# STEP 5 - Separate Features and Target
# =============================================================================
print("\n[STEP 5] Separating features (X) and target (y) ...")

X = df.drop(columns=["Churn"])
y = df["Churn"]
print(f"  Features shape : {X.shape}")
print(f"  Target shape   : {y.shape}")
print(f"  Feature columns: {list(X.columns)}")

# =============================================================================
# STEP 6 - Train / Test Split
# =============================================================================
print("\n[STEP 6] Splitting data into train/test sets (80/20 split) ...")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_test_display = X_test_raw.copy()

print(f"  Training set   : {X_train_raw.shape}")
print(f"  Testing  set   : {X_test_raw.shape}")
print(f"  Train churn distribution: {dict(y_train.value_counts().sort_index())}")
print(f"  Test  churn distribution: {dict(y_test.value_counts().sort_index())}")

# =============================================================================
# STEP 7 - Preprocessing: Encode Categoricals + Scale Numerics
# =============================================================================
print("\n[STEP 7] Preprocessing features ...")

categorical_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
numeric_cols     = X_train_raw.select_dtypes(exclude=["object"]).columns.tolist()

print(f"  Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"  Numeric columns     ({len(numeric_cols)}): {numeric_cols}")

X_train = X_train_raw.copy()
X_test  = X_test_raw.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col]  = le.transform(X_test[col].astype(str))
    label_encoders[col] = le
    print(f"  Label-encoded '{col}': {list(le.classes_)}")

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])
print("  [OK] Numeric columns scaled with StandardScaler.")

# =============================================================================
# STEP 8 - Train Models
# =============================================================================
print("\n[STEP 8] Training models ...")

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
print("  [OK] Random Forest trained  (200 trees, max_depth=10)")

xgb_model = XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss", verbosity=0
)
xgb_model.fit(X_train, y_train)
print("  [OK] XGBoost trained  (200 trees, lr=0.1, max_depth=5)")

# =============================================================================
# STEP 9 - Evaluate Both Models
# =============================================================================
print("\n[STEP 9] Evaluating models ...")


def evaluate_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    f1   = f1_score(y_te, y_pred, zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)
    print(f"\n  {'-'*52}")
    print(f"  Model : {name}")
    print(f"  {'-'*52}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred,
                                target_names=["No Churn", "Churned"],
                                zero_division=0))
    return {
        "name": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "cm": cm, "y_pred": y_pred
    }


rf_results  = evaluate_model("Random Forest", rf_model,  X_test, y_test)
xgb_results = evaluate_model("XGBoost",       xgb_model, X_test, y_test)

# =============================================================================
# STEP 10 - Compare Both Models + Identify the Winner
# =============================================================================
print("\n[STEP 10] Model Comparison ...")

metrics_df = pd.DataFrame([
    {k: v for k, v in rf_results.items()  if k not in ("cm", "y_pred")},
    {k: v for k, v in xgb_results.items() if k not in ("cm", "y_pred")},
]).set_index("name")

print(f"\n  {'Model':<22} | {'Accuracy':>9} | {'Precision':>9} | "
      f"{'Recall':>7} | {'F1-Score':>8}")
print(f"  {'-'*62}")
for idx, row in metrics_df.iterrows():
    print(f"  {idx:<22} | {row['accuracy']:>9.4f} | {row['precision']:>9.4f} | "
          f"{row['recall']:>7.4f} | {row['f1']:>8.4f}")

best_model_name = metrics_df["f1"].idxmax()
best_model      = rf_model if best_model_name == "Random Forest" else xgb_model
print(f"\n  [WINNER] Best model : {best_model_name}")
print(f"           F1-Score  : {metrics_df.loc[best_model_name, 'f1']:.4f}")

print("\n  Generating comparison plots ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison — Random Forest vs XGBoost",
             fontsize=14, fontweight="bold")

metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_keys   = ["accuracy", "precision", "recall", "f1"]
x     = np.arange(len(metric_keys))
width = 0.35

ax = axes[0]
bars_rf  = ax.bar(x - width / 2,
                  [metrics_df.loc["Random Forest", k] for k in metric_keys],
                  width, label="Random Forest", color="#2196F3", edgecolor="white")
bars_xgb = ax.bar(x + width / 2,
                  [metrics_df.loc["XGBoost", k] for k in metric_keys],
                  width, label="XGBoost", color="#F44336", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(metric_labels)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Performance Metrics", fontweight="bold")
ax.legend()
for bar in list(bars_rf) + list(bars_xgb):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=8)

ax = axes[1]
sns.heatmap(rf_results["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churned"],
            yticklabels=["No Churn", "Churned"],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix\n(Random Forest)", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax = axes[2]
sns.heatmap(xgb_results["cm"], annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Churn", "Churned"],
            yticklabels=["No Churn", "Churned"],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix\n(XGBoost)", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("model_comparison_churn.png", bbox_inches="tight")
print("  [OK] Comparison plot saved to 'model_comparison_churn.png'")
plt.close()

feat_names = list(X_train.columns)
n_show     = min(12, len(feat_names))

rf_imp  = pd.Series(rf_model.feature_importances_,  index=feat_names).sort_values()
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feat_names).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Feature Importance — Random Forest vs XGBoost",
             fontsize=14, fontweight="bold")
rf_imp.tail(n_show).plot(kind="barh", ax=axes[0], color="#2196F3", edgecolor="white")
axes[0].set_title(f"Random Forest — Top {n_show} Features", fontweight="bold")
axes[0].set_xlabel("Importance")
xgb_imp.tail(n_show).plot(kind="barh", ax=axes[1], color="#F44336", edgecolor="white")
axes[1].set_title(f"XGBoost — Top {n_show} Features", fontweight="bold")
axes[1].set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_churn.png", bbox_inches="tight")
print("  [OK] Feature importance plot saved to 'feature_importance_churn.png'")
plt.close()

# =============================================================================
# STEP 11 - Show Which Customers Are Predicted to Churn
# =============================================================================
print("\n" + "=" * 65)
print("  STEP 11 — Which Customers Are Predicted to Churn?")
print("=" * 65)

best_preds = best_model.predict(X_test)
best_proba = best_model.predict_proba(X_test)[:, 1]

predicted_churn_n  = int(best_preds.sum())
predicted_retain_n = len(best_preds) - predicted_churn_n
actual_churn_n     = int(y_test.sum())

print(f"\n  Model used              : {best_model_name}")
print(f"  Test set size           : {len(y_test):,} customers")
print(f"  Actual churners         : {actual_churn_n:,}")
print(f"  Predicted to CHURN      : {predicted_churn_n:,}")
print(f"  Predicted to RETAIN     : {predicted_retain_n:,}")

top10_idx = np.argsort(best_proba)[::-1][:10]

print(f"\n  Top 10 highest-risk customers (from test set):")
print(f"  {'-'*72}")
print(f"  {'#':<4} {'Tenure':>7} {'DaysSince':>10} {'Tier':<12} {'Purchases':>10} {'Churn Prob':>11}")
print(f"  {'-'*72}")
for rank, idx in enumerate(top10_idx, 1):
    row  = X_test_display.iloc[idx]
    prob = best_proba[idx]
    print(f"  {rank:<4} {int(row['TenureMonths']):>7} "
          f"{int(row['DaysSinceLastPurchase']):>10} "
          f"{str(row['SubscriptionTier']):<12} "
          f"{int(row['TotalPurchases']):>10} "
          f"{prob:>10.2%}")

# =============================================================================
# MODEL TESTING - Predict on Custom Customer Profiles
# =============================================================================
print("\n" + "=" * 65)
print("  MODEL TESTING - Custom Customer Churn Prediction")
print("=" * 65)

custom_customers = pd.DataFrame([
    {
        "Age": 35, "Gender": "Male", "TenureMonths": 48,
        "TotalPurchases": 85, "AvgOrderValue": 120.0,
        "DaysSinceLastPurchase": 5, "NumComplaints": 0,
        "EmailOpenRate": 60.0, "LoyaltyPoints": 7500,
        "ReturnsRate": 5.0, "SubscriptionTier": "Premium",
        "PreferredCategory": "Electronics", "PaymentMethod": "Credit Card",
    },
    {
        "Age": 22, "Gender": "Female", "TenureMonths": 2,
        "TotalPurchases": 3, "AvgOrderValue": 45.0,
        "DaysSinceLastPurchase": 150, "NumComplaints": 4,
        "EmailOpenRate": 8.0, "LoyaltyPoints": 100,
        "ReturnsRate": 35.0, "SubscriptionTier": "Free",
        "PreferredCategory": "Clothing", "PaymentMethod": "UPI",
    },
    {
        "Age": 45, "Gender": "Male", "TenureMonths": 24,
        "TotalPurchases": 40, "AvgOrderValue": 200.0,
        "DaysSinceLastPurchase": 45, "NumComplaints": 1,
        "EmailOpenRate": 35.0, "LoyaltyPoints": 3000,
        "ReturnsRate": 10.0, "SubscriptionTier": "Basic",
        "PreferredCategory": "Home", "PaymentMethod": "Debit Card",
    },
    {
        "Age": 28, "Gender": "Female", "TenureMonths": 1,
        "TotalPurchases": 1, "AvgOrderValue": 30.0,
        "DaysSinceLastPurchase": 200, "NumComplaints": 5,
        "EmailOpenRate": 5.0, "LoyaltyPoints": 50,
        "ReturnsRate": 45.0, "SubscriptionTier": "Free",
        "PreferredCategory": "Sports", "PaymentMethod": "UPI",
    },
    {
        "Age": 55, "Gender": "Male", "TenureMonths": 56,
        "TotalPurchases": 130, "AvgOrderValue": 300.0,
        "DaysSinceLastPurchase": 10, "NumComplaints": 0,
        "EmailOpenRate": 75.0, "LoyaltyPoints": 9500,
        "ReturnsRate": 3.0, "SubscriptionTier": "Premium",
        "PreferredCategory": "Electronics", "PaymentMethod": "Credit Card",
    },
])

customer_labels  = [
    "Cust A — 48-month tenure, Premium,  85 purchases, 5 days ago",
    "Cust B — 2-month tenure,  Free,     3 purchases,  150 days ago",
    "Cust C — 24-month tenure, Basic,    40 purchases, 45 days ago",
    "Cust D — 1-month tenure,  Free,     1 purchase,   200 days ago",
    "Cust E — 56-month tenure, Premium, 130 purchases, 10 days ago",
]
expected_results = ["No Churn", "Churn", "Moderate", "Churn", "No Churn"]

custom_enc = custom_customers.copy()
for col in categorical_cols:
    le = label_encoders[col]
    custom_enc[col] = custom_enc[col].astype(str).apply(
        lambda x, le=le: int(le.transform([x])[0])
        if x in le.classes_ else 0
    )

custom_enc[numeric_cols] = scaler.transform(custom_enc[numeric_cols])
custom_enc = custom_enc[X_train.columns]

rf_cust_preds  = rf_model.predict(custom_enc)
xgb_cust_preds = xgb_model.predict(custom_enc)
rf_cust_proba  = rf_model.predict_proba(custom_enc)[:, 1]
xgb_cust_proba = xgb_model.predict_proba(custom_enc)[:, 1]

print(f"\n  {'Customer':<60} {'RF':>10} {'XGB':>10}")
print(f"  {'-'*84}")
for i, label in enumerate(customer_labels):
    rf_lbl  = "CHURN" if rf_cust_preds[i]  == 1 else "Stay"
    xgb_lbl = "CHURN" if xgb_cust_preds[i] == 1 else "Stay"
    print(f"  {label:<60} "
          f"{rf_lbl+' ('+f'{rf_cust_proba[i]:.0%}'+')':>10}  "
          f"{xgb_lbl+' ('+f'{xgb_cust_proba[i]:.0%}'+')':>10}")

print(f"\n  {'='*65}")
print("  DETAILED PREDICTIONS PER CUSTOMER")
print(f"  {'='*65}")

for i, (label, exp) in enumerate(zip(customer_labels, expected_results)):
    rf_pred   = rf_cust_preds[i]
    xgb_pred  = xgb_cust_preds[i]
    rf_label  = "WILL CHURN" if rf_pred  == 1 else "Will Stay"
    xgb_label = "WILL CHURN" if xgb_pred == 1 else "Will Stay"

    print(f"\n  {label}")
    print(f"  {'-'*60}")
    print(f"  Random Forest : {rf_label:<12} (churn probability: {rf_cust_proba[i]:.2%})")
    print(f"  XGBoost       : {xgb_label:<12} (churn probability: {xgb_cust_proba[i]:.2%})")
    print(f"  Expected      : {exp}")

    if rf_pred == xgb_pred:
        print("  Agreement     : Both models agree.")
    else:
        print("  Agreement     : Models disagree.")

agree_pct = sum(rf_cust_preds == xgb_cust_preds) / len(customer_labels) * 100
print(f"\n  Model agreement on custom profiles : {agree_pct:.0f}%")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Dataset          : {len(df):,} e-commerce customers")
print(f"  No Churn         : {churn_counts.get(0,0):,} ({churn_pct.get(0,0):.2f}%)")
print(f"  Churned          : {churn_counts.get(1,0):,} ({churn_pct.get(1,0):.2f}%)")
print(f"  {'─'*45}")
print(f"  {'Model':<22} | {'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
print(f"  {'─'*55}")
for idx, row in metrics_df.iterrows():
    star = " *" if idx == best_model_name else ""
    print(f"  {idx:<22} | {row['accuracy']:>7.4f} | "
          f"{row['precision']:>7.4f} | {row['recall']:>7.4f} | "
          f"{row['f1']:>7.4f}{star}")
print(f"  {'─'*55}")
print(f"\n  [WINNER] Best Model : {best_model_name}  *")
print(f"           F1-Score   : {metrics_df.loc[best_model_name, 'f1']:.4f}")
print("=" * 65)
print("\n  Output files:")
print("    eda_churn_plots.png          — Exploratory Data Analysis")
print("    model_comparison_churn.png   — Metrics comparison & Confusion matrices")
print("    feature_importance_churn.png — Feature importance (RF & XGBoost)")
print("=" * 65)
