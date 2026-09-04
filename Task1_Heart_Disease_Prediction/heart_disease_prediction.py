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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("=" * 65)
print("Task 1: Heart Disease Prediction")
print("=" * 65)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")

print("\n[STEP 1] Loading dataset from UCI Repository ...")
try:
    df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
    print(f"  [OK] Dataset loaded successfully  ->  Shape: {df.shape}")
except Exception as e:
    print(f"  [WARN] Remote load failed ({e}).")
    print("  [INFO] Generating a synthetic fallback dataset ...")

    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        "age":      np.random.randint(29, 77, n),
        "sex":      np.random.randint(0, 2, n),
        "cp":       np.random.randint(0, 4, n),
        "trestbps": np.random.randint(94, 200, n),
        "chol":     np.random.randint(126, 564, n),
        "fbs":      np.random.randint(0, 2, n),
        "restecg":  np.random.randint(0, 3, n),
        "thalach":  np.random.randint(71, 202, n),
        "exang":    np.random.randint(0, 2, n),
        "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
        "slope":    np.random.randint(0, 3, n),
        "ca":       np.random.choice([0, 1, 2, 3, np.nan], n),
        "thal":     np.random.choice([0, 1, 2, np.nan], n),
        "target":   np.random.randint(0, 2, n),
    })
    print(f"  [OK] Synthetic dataset generated  ->  Shape: {df.shape}")

print("\n[STEP 2] Inspecting the dataset ...")
print(f"\n  First 5 rows:\n{df.head().to_string()}")
print(f"\n  Dataset shape  : {df.shape}")
print(f"\n  Column dtypes  :\n{df.dtypes.to_string()}")
print(f"\n  Basic statistics:\n{df.describe().to_string()}")
print(f"\n  Missing values per column:\n{df.isnull().sum().to_string()}")
print(f"\n  Duplicate rows  : {df.duplicated().sum()}")

print("\n[STEP 3] Cleaning the data ...")

df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

before_dup = len(df)
df.drop_duplicates(inplace=True)
print(f"  Duplicates removed : {before_dup - len(df)}")

for col in ["ca", "thal"]:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"  Missing '{col}' filled with mode = {mode_val}")

assert df.isnull().sum().sum() == 0, "There are still missing values!"
print("  [OK] No missing values remaining.")
print(f"  Dataset shape after cleaning : {df.shape}")

print("\n[STEP 4] Performing basic data analysis ...")

print(f"\n  Target distribution:\n{df['target'].value_counts().to_string()}")
print(f"  Target proportions:\n"
      f"{df['target'].value_counts(normalize=True).mul(100).round(2).to_string()} %")
print(f"\n  Mean values grouped by target:\n{df.groupby('target').mean().round(2).to_string()}")

print("\n  Generating EDA plots ...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Heart Disease Dataset - Exploratory Data Analysis",
             fontsize=16, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
counts = df["target"].value_counts().sort_index()
ax1.bar(["No Disease (0)", "Disease (1)"], counts.values,
        color=["#2196F3", "#F44336"], edgecolor="white", linewidth=0.8)
ax1.set_title("Target Distribution", fontweight="bold")
ax1.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax1.text(i, v + 2, str(v), ha="center", fontweight="bold")

ax2 = fig.add_subplot(gs[0, 1])
for tgt, color, label in zip([0, 1], ["#2196F3", "#F44336"],
                              ["No Disease", "Disease"]):
    ax2.hist(df[df["target"] == tgt]["age"], bins=15, alpha=0.7,
             color=color, label=label, edgecolor="white")
ax2.set_title("Age Distribution by Target", fontweight="bold")
ax2.set_xlabel("Age")
ax2.set_ylabel("Count")
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
for tgt, color, label in zip([0, 1], ["#2196F3", "#F44336"],
                              ["No Disease", "Disease"]):
    ax3.hist(df[df["target"] == tgt]["chol"], bins=20, alpha=0.7,
             color=color, label=label, edgecolor="white")
ax3.set_title("Cholesterol Distribution by Target", fontweight="bold")
ax3.set_xlabel("Cholesterol (mg/dl)")
ax3.set_ylabel("Count")
ax3.legend()

ax4 = fig.add_subplot(gs[1, 0])
sex_target = df.groupby(["sex", "target"]).size().unstack(fill_value=0)
sex_target.index = ["Female", "Male"]
sex_target.columns = ["No Disease", "Disease"]
sex_target.plot(kind="bar", ax=ax4, color=["#2196F3", "#F44336"],
                edgecolor="white", rot=0)
ax4.set_title("Gender vs Heart Disease", fontweight="bold")
ax4.set_ylabel("Count")
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
cp_counts.columns = ["No Disease", "Disease"]
cp_counts.plot(kind="bar", ax=ax5, color=["#2196F3", "#F44336"],
               edgecolor="white", rot=0)
ax5.set_title("Chest Pain Type vs Heart Disease", fontweight="bold")
ax5.set_xlabel("Chest Pain Type")
ax5.set_ylabel("Count")
ax5.legend()

ax6 = fig.add_subplot(gs[1, 2])
for tgt, color, label in zip([0, 1], ["#2196F3", "#F44336"],
                              ["No Disease", "Disease"]):
    ax6.hist(df[df["target"] == tgt]["thalach"], bins=20, alpha=0.7,
             color=color, label=label, edgecolor="white")
ax6.set_title("Max Heart Rate (thalach) by Target", fontweight="bold")
ax6.set_xlabel("Max Heart Rate")
ax6.set_ylabel("Count")
ax6.legend()

ax7 = fig.add_subplot(gs[2, :])
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax7, cbar_kws={"shrink": 0.6})
ax7.set_title("Feature Correlation Heatmap", fontweight="bold")

plt.savefig("eda_plots.png", bbox_inches="tight")
print("  [OK] EDA plots saved to 'eda_plots.png'")
plt.close()

print("\n[STEP 5] Separating features (X) and target (y) ...")
X = df.drop(columns=["target"])
y = df["target"]
print(f"  Features shape : {X.shape}")
print(f"  Target shape   : {y.shape}")
print(f"  Feature columns: {list(X.columns)}")

print("\n[STEP 6] Splitting data into train/test sets (80/20 split) ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training set   : {X_train.shape}")
print(f"  Testing  set   : {X_test.shape}")
print(f"  Train target distribution: {dict(y_train.value_counts())}")
print(f"  Test  target distribution: {dict(y_test.value_counts())}")

print("\n[STEP 7] Applying feature scaling (StandardScaler) ...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("  [OK] Features scaled. Mean ~= 0, Std ~= 1 after scaling.")

print("\n[STEP 8] Training models ...")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
print("  [OK] Logistic Regression trained.")

svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
svm_model.fit(X_train_scaled, y_train)
print("  [OK] SVM (RBF kernel) trained.")

print("\n[STEP 9] Evaluating models ...")


def evaluate_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec  = recall_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)

    print(f"\n  {'-'*50}")
    print(f"  Model : {name}")
    print(f"  {'-'*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred,
                                target_names=["No Disease", "Disease"]))

    return {
        "name": name, "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "cm": cm, "y_pred": y_pred
    }


lr_results  = evaluate_model("Logistic Regression", lr_model,  X_test_scaled, y_test)
svm_results = evaluate_model("SVM (RBF Kernel)",    svm_model, X_test_scaled, y_test)

print("\n[STEP 10] Model Comparison ...")

metrics_df = pd.DataFrame([
    {k: v for k, v in lr_results.items()  if k not in ("cm", "y_pred")},
    {k: v for k, v in svm_results.items() if k not in ("cm", "y_pred")},
]).set_index("name")

print(f"\n{metrics_df.to_string()}")

best_model = metrics_df["f1"].idxmax()
print(f"\n  [WINNER] Better Model : {best_model}")
print(f"           (F1-score: {metrics_df.loc[best_model, 'f1']:.4f})")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Comparison - Logistic Regression vs SVM",
             fontsize=14, fontweight="bold")

metric_cols = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metric_cols))
width = 0.35

ax = axes[0]
bars1 = ax.bar(x - width/2,
               metrics_df.loc["Logistic Regression", metric_cols],
               width, label="Logistic Regression", color="#2196F3", edgecolor="white")
bars2 = ax.bar(x + width/2,
               metrics_df.loc["SVM (RBF Kernel)", metric_cols],
               width, label="SVM (RBF Kernel)", color="#F44336", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Performance Metrics Comparison", fontweight="bold")
ax.legend()
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)

ax = axes[1]
sns.heatmap(lr_results["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix\n(Logistic Regression)", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax = axes[2]
sns.heatmap(svm_results["cm"], annot=True, fmt="d", cmap="Reds",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"],
            ax=ax, linewidths=0.5)
ax.set_title("Confusion Matrix\n(SVM - RBF Kernel)", fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("model_comparison.png", bbox_inches="tight")
print("\n  [OK] Comparison plot saved to 'model_comparison.png'")
plt.close()

print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
for idx, row in metrics_df.iterrows():
    print(f"  {idx:<25} | Acc: {row['accuracy']:.4f} | Prec: {row['precision']:.4f}"
          f" | Rec: {row['recall']:.4f} | F1: {row['f1']:.4f}")
print(f"\n  [WINNER] Recommended Model : {best_model}")
print("=" * 65)

print("\n" + "=" * 65)
print("  MODEL TESTING - Predicting on New Patient Samples")
print("=" * 65)

test_patients = pd.DataFrame([
    {
        "age": 35, "sex": 0, "cp": 1,  "trestbps": 118, "chol": 195,
        "fbs": 0,  "restecg": 0, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    },
    {
        "age": 65, "sex": 1, "cp": 4,  "trestbps": 160, "chol": 320,
        "fbs": 1,  "restecg": 2, "thalach": 105, "exang": 1,
        "oldpeak": 3.5, "slope": 2, "ca": 3, "thal": 7
    },
    {
        "age": 52, "sex": 1, "cp": 3,  "trestbps": 135, "chol": 255,
        "fbs": 0,  "restecg": 1, "thalach": 142, "exang": 0,
        "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 3
    },
    {
        "age": 58, "sex": 1, "cp": 4,  "trestbps": 150, "chol": 286,
        "fbs": 0,  "restecg": 2, "thalach": 108, "exang": 1,
        "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3
    },
    {
        "age": 29, "sex": 0, "cp": 1,  "trestbps": 112, "chol": 182,
        "fbs": 0,  "restecg": 0, "thalach": 185, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3
    },
])

patient_labels = [
    "Patient A (Low-risk female, age 35)",
    "Patient B (High-risk male, age 65)",
    "Patient C (Moderate-risk male, age 52)",
    "Patient D (High-risk male, age 58)",
    "Patient E (Low-risk female, age 29)",
]

test_patients_scaled = scaler.transform(test_patients)

lr_test_preds   = lr_model.predict(test_patients_scaled)
svm_test_preds  = svm_model.predict(test_patients_scaled)

lr_test_proba   = lr_model.predict_proba(test_patients_scaled)

print(f"\n  {'Patient':<40} {'LR Prediction':<20} {'LR Confidence':<18} {'SVM Prediction'}")
print(f"  {'-'*95}")

for i, label in enumerate(patient_labels):
    lr_pred  = "DISEASE" if lr_test_preds[i]  == 1 else "No Disease"
    svm_pred = "DISEASE" if svm_test_preds[i] == 1 else "No Disease"

    lr_conf  = lr_test_proba[i][1] * 100

    lr_flag  = " <-- HIGH RISK" if lr_test_preds[i]  == 1 else ""
    svm_flag = " <-- HIGH RISK" if svm_test_preds[i] == 1 else ""

    print(f"  {label:<40} {lr_pred:<20} {lr_conf:>6.2f}%           {svm_pred}{svm_flag}")

print(f"\n  {'='*65}")
print("  DETAILED PATIENT PREDICTIONS")
print(f"  {'='*65}")

for i, label in enumerate(patient_labels):
    print(f"\n  {label}")
    print(f"  {'-'*55}")

    row = test_patients.iloc[i]
    print(f"  Features  : age={int(row.age)}, sex={'M' if row.sex==1 else 'F'}, "
          f"cp={int(row.cp)}, BP={int(row.trestbps)}, chol={int(row.chol)}, "
          f"thalach={int(row.thalach)}, oldpeak={row.oldpeak}, ca={int(row.ca)}")

    lr_pred   = lr_test_preds[i]
    lr_label  = "HEART DISEASE DETECTED" if lr_pred == 1 else "No Heart Disease"
    lr_no_dis = lr_test_proba[i][0] * 100
    lr_dis    = lr_test_proba[i][1] * 100
    print(f"  LR Result : {lr_label}")
    print(f"             Probability -> No Disease: {lr_no_dis:.2f}%  |  Disease: {lr_dis:.2f}%")

    svm_pred  = svm_test_preds[i]
    svm_label = "HEART DISEASE DETECTED" if svm_pred == 1 else "No Heart Disease"
    print(f"  SVM Result: {svm_label}")

    if lr_pred == svm_pred:
        print("  Agreement : Both models AGREE.")
    else:
        print("  Agreement : Models DISAGREE - consult clinical judgment.")

agreements   = sum(lr_test_preds == svm_test_preds)
total_tested = len(patient_labels)
print(f"\n  Model Agreement Rate on test patients: "
      f"{agreements}/{total_tested} ({agreements/total_tested*100:.0f}%)")

print("\n" + "=" * 65)
print("  Done! Output files: eda_plots.png  |  model_comparison.png")
print("=" * 65)
