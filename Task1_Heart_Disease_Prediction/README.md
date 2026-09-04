# Task 1: Healthcare – Heart Disease Prediction (Classification)
### Growfinix Machine Learning Internship

---

## 📌 Objective

Build a binary classification model to predict whether a patient has heart disease based on clinical features such as **age, blood pressure, cholesterol, chest pain type, and other health metrics**.

Two models are trained and compared:
- **Logistic Regression**
- **Support Vector Machine (SVM)** with RBF kernel

---

## 📁 Project Structure

```
Task1_Heart_Disease_Prediction/
│
├── heart_disease_prediction.py   # Main Python script (all steps)
├── eda_plots.png                 # EDA visualisations (auto-generated)
├── model_comparison.png          # Model comparison chart (auto-generated)
└── README.md                     # This file
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository – Heart Disease (Cleveland)](https://archive.ics.uci.edu/ml/datasets/heart+disease) |
| **Rows** | 303 patient records |
| **Features** | 13 clinical attributes |
| **Target** | Binary (0 = No Disease, 1 = Disease) |

### Feature Descriptions

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = Male, 0 = Female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia type (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect) |

---

## 🚀 Steps Performed

| Step | Description |
|---|---|
| 1 | Load dataset from UCI Repository (with synthetic fallback) |
| 2 | Inspect shape, dtypes, statistics |
| 3 | Clean data — binarise target, remove duplicates |
| 4 | Handle missing values (ca, thal) using mode imputation |
| 5 | Exploratory Data Analysis (EDA) with 7 plots |
| 6 | Separate features (X) and target (y) |
| 7 | 80/20 stratified train/test split |
| 8 | Feature scaling with StandardScaler |
| 9 | Train Logistic Regression and SVM models |
| 10 | Evaluate using accuracy, precision, recall, F1, confusion matrix, classification report |
| 11 | Compare models and identify the better performer |

---

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Requires Python 3.8+

---

## How to Run

```bash
# Navigate to the project folder
cd Task1_Heart_Disease_Prediction

# Run the script
python heart_disease_prediction.py
```

The script will:
1. Automatically download the UCI Cleveland Heart Disease dataset.
2. If the download fails (no internet), generate a synthetic fallback dataset.
3. Produce all outputs in the terminal and save two plot images.

---

## Output Files

| File | Description |
|---|---|
| `eda_plots.png` | Target distribution, age/cholesterol histograms, gender breakdown, chest-pain type, max heart rate, correlation heatmap |
| `model_comparison.png` | Side-by-side bar chart of all metrics + confusion matrices for both models |

---

## Evaluation Metrics

Each model is evaluated on the test set using:

- **Accuracy** — Overall correct predictions
- **Precision** — Of all positive predictions, how many were correct?
- **Recall** — Of all actual positives, how many were correctly identified?
- **F1-Score** — Harmonic mean of Precision and Recall (primary metric)
- **Confusion Matrix** — TP, TN, FP, FN breakdown
- **Classification Report** — Per-class precision, recall, F1, support

Winner is selected by F1-Score (more robust than accuracy on medical datasets).

---

## Key Design Decisions

- **Stratified split** — ensures both train and test sets have proportional class distribution
- **StandardScaler** is fit only on training data to prevent data leakage into the test set
- **SVM with RBF kernel** — performs better than linear SVM on non-linearly separable medical data
- **Mode imputation** — used for the two features (ca, thal) with a handful of missing values

---

## About

**Internship**: Growfinix Machine Learning Internship
**Task**: Task 1 – Healthcare – Heart Disease Prediction (Classification)
**Domain**: Healthcare / Binary Classification
**Tools**: Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
