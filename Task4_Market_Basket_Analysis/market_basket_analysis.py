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
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    print("[ERROR] mlxtend not found.")
    print("        Run: pip install mlxtend")
    sys.exit(1)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("=" * 65)
print("  GROWFINIX ML INTERNSHIP - Task 4: Market Basket Analysis")
print("=" * 65)

print("\n[STEP 1] Generating supermarket transaction dataset ...")

np.random.seed(42)
N_TRANSACTIONS = 2000

products = [
    "Bread", "Butter", "Milk", "Eggs", "Cheese",
    "Yogurt", "Chicken", "Fish", "Rice", "Pasta",
    "Tomatoes", "Onions", "Potatoes", "Lettuce", "Carrots",
    "Apples", "Bananas", "Oranges", "Grapes", "Strawberries",
    "Coffee", "Tea", "Juice", "Soda", "Water",
    "Chips", "Cookies", "Chocolate", "Cake", "Soap",
]

def make_basket_probs():
    probs = np.random.uniform(0.05, 0.30, len(products))
    probs[products.index("Bread")]       *= 2.5
    probs[products.index("Milk")]        *= 2.2
    probs[products.index("Eggs")]        *= 2.0
    probs[products.index("Bananas")]     *= 1.8
    probs[products.index("Rice")]        *= 1.7
    return np.clip(probs, 0.05, 0.70)

base_probs = make_basket_probs()

def generate_basket(probs):
    chosen = [p for p, prob in zip(products, probs) if np.random.random() < prob]
    if not chosen:
        chosen = [np.random.choice(products)]
    return chosen

raw_transactions = []
for i in range(N_TRANSACTIONS):
    basket = generate_basket(base_probs)
    raw_transactions.append(basket)

df_raw = pd.DataFrame({
    "TransactionID": [f"T{i+1:04d}" for i in range(N_TRANSACTIONS)],
    "Items":         [", ".join(b) for b in raw_transactions],
})

print(f"  [OK] Dataset generated -> {df_raw.shape[0]} transactions, {len(products)} products")
print(f"\n  First 5 transactions:")
print(df_raw.head(5).to_string(index=False))

print("\n[STEP 2] Inspecting the dataset ...")

basket_sizes = [len(b) for b in raw_transactions]
print(f"\n  Total transactions  : {len(df_raw):,}")
print(f"  Unique products     : {len(products)}")
print(f"  Avg items per basket: {np.mean(basket_sizes):.2f}")
print(f"  Min items in basket : {min(basket_sizes)}")
print(f"  Max items in basket : {max(basket_sizes)}")
print(f"\n  Column dtypes:\n{df_raw.dtypes.to_string()}")
print(f"\n  Missing values : {df_raw.isnull().sum().sum()}")
print(f"  Duplicate rows : {df_raw.duplicated().sum()}")

print("\n[STEP 3] Cleaning the transaction data ...")

before = len(df_raw)
df_raw.drop_duplicates(inplace=True)
print(f"  Duplicate rows removed : {before - len(df_raw)}")

missing = df_raw.isnull().sum().sum()
if missing > 0:
    df_raw.dropna(inplace=True)
    print(f"  Rows with NaN dropped  : {missing}")
else:
    print("  No missing values found.")

df_raw["Items"] = df_raw["Items"].str.strip()
df_raw = df_raw[df_raw["Items"].str.len() > 0]
print(f"  [OK] Data is clean. Transactions remaining: {len(df_raw):,}")

print("\n[STEP 4] Preparing transaction data for market basket analysis ...")

transactions = [row.split(", ") for row in df_raw["Items"]]
transactions = [[item.strip() for item in basket if item.strip()] for basket in transactions]

item_counts = {}
for basket in transactions:
    for item in basket:
        item_counts[item] = item_counts.get(item, 0) + 1

item_freq = pd.Series(item_counts).sort_values(ascending=False)
print(f"\n  Top 15 most purchased products:")
print(f"  {'Product':<15} {'Count':>8} {'Support %':>10}")
print(f"  {'-'*38}")
for item, count in item_freq.head(15).items():
    supp = count / len(transactions) * 100
    print(f"  {item:<15} {count:>8,} {supp:>9.2f}%")

print("\n[STEP 5] Converting to binary (one-hot encoded) format ...")

te     = TransactionEncoder()
te_arr = te.fit(transactions).transform(transactions)
df_enc = pd.DataFrame(te_arr, columns=te.columns_)

print(f"  [OK] Encoded matrix shape: {df_enc.shape}")
print(f"  Columns (products): {list(df_enc.columns)}")
print(f"\n  Sample (first 3 rows, first 8 columns):")
print(df_enc.iloc[:3, :8].to_string())
print(f"\n  Item purchase rates (support):")
item_support = df_enc.mean().sort_values(ascending=False)
print(f"  {'Product':<15} {'Support':>8}")
print(f"  {'-'*26}")
for item, supp in item_support.head(10).items():
    print(f"  {item:<15} {supp:>8.4f}  ({supp*100:.2f}%)")

MIN_SUPPORT    = 0.05
MIN_CONFIDENCE = 0.20
MIN_LIFT       = 1.0

print(f"\n[STEP 6] Running Apriori algorithm ...")
print(f"  min_support    : {MIN_SUPPORT}  ({MIN_SUPPORT*100:.0f}% of transactions)")
print(f"  min_confidence : {MIN_CONFIDENCE} (for rules — Step 7)")
print(f"  min_lift       : {MIN_LIFT} (for rule filtering — Step 7)")

frequent_itemsets = apriori(
    df_enc,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=4,
)
frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False).reset_index(drop=True)

n_1 = len(frequent_itemsets[frequent_itemsets["length"] == 1])
n_2 = len(frequent_itemsets[frequent_itemsets["length"] == 2])
n_3 = len(frequent_itemsets[frequent_itemsets["length"] == 3])
n_4 = len(frequent_itemsets[frequent_itemsets["length"] >= 4])

print(f"\n  Frequent itemsets found : {len(frequent_itemsets):,}")
print(f"  Single-item  (1): {n_1}")
print(f"  Two-item     (2): {n_2}")
print(f"  Three-item   (3): {n_3}")
print(f"  Four-item+  (4+): {n_4}")

print(f"\n  Top 15 frequent itemsets by support:")
print(f"  {'Itemset':<40} {'Support':>8} {'Length':>7}")
print(f"  {'-'*58}")
for _, row in frequent_itemsets.head(15).iterrows():
    items_str = " + ".join(sorted(row["itemsets"]))
    print(f"  {items_str:<40} {row['support']:>8.4f} {row['length']:>7}")

print(f"\n[STEP 7] Generating association rules (metric=lift, min_threshold={MIN_LIFT}) ...")

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=MIN_LIFT,
)

rules = rules[rules["confidence"] >= MIN_CONFIDENCE].copy()
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

print(f"  [OK] Association rules generated: {len(rules):,}")

print("\n[STEP 8] Support, Confidence, and Lift — statistics overview ...")

print(f"\n  Metric       {'Min':>8} {'Mean':>8} {'Max':>8}")
print(f"  {'-'*38}")
for metric in ["support", "confidence", "lift"]:
    print(f"  {metric.capitalize():<12} "
          f"{rules[metric].min():>8.4f} "
          f"{rules[metric].mean():>8.4f} "
          f"{rules[metric].max():>8.4f}")

print(f"\n  Top 20 rules by LIFT  (sorted highest → lowest):")
print(f"  {'#':<3} {'Antecedent':<25} {'Consequent':<20} "
      f"{'Support':>9} {'Conf':>7} {'Lift':>7}")
print(f"  {'-'*75}")
for i, row in rules.head(20).iterrows():
    ant = " + ".join(sorted(row["antecedents"]))
    con = " + ".join(sorted(row["consequents"]))
    print(f"  {i+1:<3} {ant:<25} {con:<20} "
          f"{row['support']:>9.4f} {row['confidence']:>7.4f} {row['lift']:>7.4f}")

print("\n" + "=" * 65)
print("  STEP 9 — Product Relationships & Purchasing Patterns")
print("=" * 65)

strong_rules = rules[(rules["lift"] >= 1.5) & (rules["confidence"] >= 0.35)].copy()
print(f"\n  STRONG rules (lift >= 1.5 AND confidence >= 35%) : {len(strong_rules)}")

if len(strong_rules) > 0:
    print(f"\n  Strong association rules:")
    print(f"  {'Antecedent':<25} => {'Consequent':<20} "
          f"{'Support':>9} {'Conf':>7} {'Lift':>7}")
    print(f"  {'-'*72}")
    for _, row in strong_rules.head(15).iterrows():
        ant = " + ".join(sorted(row["antecedents"]))
        con = " + ".join(sorted(row["consequents"]))
        print(f"  {ant:<25} => {con:<20} "
              f"{row['support']:>9.4f} {row['confidence']:>7.4f} {row['lift']:>7.4f}")

top1_rule = rules.iloc[0]
print(f"\n  Top product pair by lift:")
ant = " + ".join(sorted(top1_rule["antecedents"]))
con = " + ".join(sorted(top1_rule["consequents"]))
print(f"  '{ant}'  =>  '{con}'")
print(f"  Support    : {top1_rule['support']:.4f}  ({top1_rule['support']*100:.2f}% of all transactions)")
print(f"  Confidence : {top1_rule['confidence']:.4f}  ({top1_rule['confidence']*100:.2f}%)")
print(f"  Lift       : {top1_rule['lift']:.4f}  (customers buying '{ant}' are "
      f"{top1_rule['lift']:.1f}x more likely to also buy '{con}')")

print(f"\n  High-confidence rules (conf >= 40%):")
high_conf = rules[rules["confidence"] >= 0.40].sort_values("confidence", ascending=False)
if len(high_conf) > 0:
    print(f"  {'Antecedent':<25} => {'Consequent':<20} {'Conf':>7} {'Lift':>7}")
    print(f"  {'-'*65}")
    for _, row in high_conf.head(10).iterrows():
        ant = " + ".join(sorted(row["antecedents"]))
        con = " + ".join(sorted(row["consequents"]))
        print(f"  {ant:<25} => {con:<20} {row['confidence']:>7.4f} {row['lift']:>7.4f}")
else:
    print("  None found at this threshold.")

print(f"\n  Rules by number of antecedent items:")
for length in [1, 2, 3]:
    subset = rules[rules["antecedents"].apply(len) == length]
    print(f"  {length}-item antecedents : {len(subset):>4} rules")

print("\n[STEP 10] Generating visualisations ...")

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Market Basket Analysis — Association Rules (Apriori)",
             fontsize=16, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.32)

ax1 = fig.add_subplot(gs[0, :])
top_items = item_support.head(20)
colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(top_items)))[::-1]
bars = ax1.barh(top_items.index[::-1], top_items.values[::-1], color=colors[::-1], edgecolor="white")
ax1.set_title("Top 20 Products by Purchase Support", fontweight="bold")
ax1.set_xlabel("Support (fraction of transactions)")
for bar, val in zip(bars, top_items.values[::-1]):
    ax1.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=8)

ax2 = fig.add_subplot(gs[1, 0])
sc = ax2.scatter(rules["support"], rules["confidence"],
                 c=rules["lift"], cmap="RdYlGn", alpha=0.7,
                 s=60, edgecolors="none")
plt.colorbar(sc, ax=ax2, label="Lift")
ax2.set_title("Support vs Confidence (coloured by Lift)", fontweight="bold")
ax2.set_xlabel("Support")
ax2.set_ylabel("Confidence")
ax2.axhline(MIN_CONFIDENCE, color="gray", linestyle="--", linewidth=0.8, label=f"min_conf={MIN_CONFIDENCE}")
ax2.legend(fontsize=8)

ax3 = fig.add_subplot(gs[1, 1])
sc2 = ax3.scatter(rules["support"], rules["lift"],
                  c=rules["confidence"], cmap="coolwarm", alpha=0.7,
                  s=60, edgecolors="none")
plt.colorbar(sc2, ax=ax3, label="Confidence")
ax3.set_title("Support vs Lift (coloured by Confidence)", fontweight="bold")
ax3.set_xlabel("Support")
ax3.set_ylabel("Lift")
ax3.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Lift = 1 (random)")
ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs[2, 0])
top_rules = rules.head(15).copy()
rule_labels = [
    f"{' + '.join(sorted(r['antecedents']))} => {' + '.join(sorted(r['consequents']))}"
    for _, r in top_rules.iterrows()
]
short_labels = [l[:35] + "..." if len(l) > 35 else l for l in rule_labels]
colors_lift = plt.cm.YlOrRd(
    (top_rules["lift"].values - top_rules["lift"].min()) /
    (top_rules["lift"].max() - top_rules["lift"].min() + 1e-9)
)
bars4 = ax4.barh(range(len(top_rules))[::-1], top_rules["lift"].values,
                 color=colors_lift, edgecolor="white")
ax4.set_yticks(range(len(top_rules))[::-1])
ax4.set_yticklabels(short_labels, fontsize=7)
ax4.set_title("Top 15 Rules by Lift", fontweight="bold")
ax4.set_xlabel("Lift")
ax4.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
for bar, val in zip(bars4, top_rules["lift"].values):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=7)

ax5 = fig.add_subplot(gs[2, 1])
top_conf_rules = rules.nlargest(15, "confidence").copy()
conf_labels = [
    f"{' + '.join(sorted(r['antecedents']))} => {' + '.join(sorted(r['consequents']))}"
    for _, r in top_conf_rules.iterrows()
]
short_conf = [l[:35] + "..." if len(l) > 35 else l for l in conf_labels]
colors_conf = plt.cm.Blues(
    (top_conf_rules["confidence"].values - top_conf_rules["confidence"].min()) /
    (top_conf_rules["confidence"].max() - top_conf_rules["confidence"].min() + 1e-9) * 0.7 + 0.3
)
ax5.barh(range(len(top_conf_rules))[::-1], top_conf_rules["confidence"].values,
         color=colors_conf, edgecolor="white")
ax5.set_yticks(range(len(top_conf_rules))[::-1])
ax5.set_yticklabels(short_conf, fontsize=7)
ax5.set_title("Top 15 Rules by Confidence", fontweight="bold")
ax5.set_xlabel("Confidence")
for i, val in enumerate(top_conf_rules["confidence"].values):
    ax5.text(val + 0.003, len(top_conf_rules) - 1 - i,
             f"{val:.3f}", va="center", fontsize=7)

plt.savefig("market_basket_analysis.png", bbox_inches="tight")
print("  [OK] Main plots saved to 'market_basket_analysis.png'")
plt.close()

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle("Market Basket Analysis — Itemset Size Distribution & Metric Heatmap",
              fontsize=13, fontweight="bold")

ax_left = axes2[0]
size_counts = frequent_itemsets["length"].value_counts().sort_index()
ax_left.bar(size_counts.index.astype(str), size_counts.values,
            color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"][:len(size_counts)],
            edgecolor="white")
ax_left.set_title("Frequent Itemsets by Size", fontweight="bold")
ax_left.set_xlabel("Number of items in itemset")
ax_left.set_ylabel("Count")
for x, y in zip(size_counts.index.astype(str), size_counts.values):
    ax_left.text(x, y + 0.5, str(y), ha="center", fontweight="bold")

ax_right = axes2[1]
top20 = rules.head(20).copy().reset_index(drop=True)
heat_data = top20[["support", "confidence", "lift"]].copy()
heat_data["lift"] = heat_data["lift"] / heat_data["lift"].max()
rule_strs_short = [
    f"{' + '.join(sorted(r['antecedents'])[:1])}=>{' + '.join(sorted(r['consequents'])[:1])}"
    for _, r in top20.iterrows()
]
import seaborn as sns
sns.heatmap(heat_data.T, ax=ax_right, cmap="YlOrRd", annot=False,
            yticklabels=["Support", "Confidence", "Lift (norm)"],
            xticklabels=[f"R{i+1}" for i in range(len(top20))],
            linewidths=0.3)
ax_right.set_title("Metrics Heatmap — Top 20 Rules by Lift", fontweight="bold")
ax_right.set_xlabel("Rule")

plt.tight_layout()
plt.savefig("itemset_heatmap.png", bbox_inches="tight")
print("  [OK] Heatmap saved to 'itemset_heatmap.png'")
plt.close()

print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Transactions          : {len(transactions):,}")
print(f"  Unique products       : {len(products)}")
print(f"  Min support used      : {MIN_SUPPORT}  ({MIN_SUPPORT*100:.0f}%)")
print(f"  Min confidence used   : {MIN_CONFIDENCE}  ({MIN_CONFIDENCE*100:.0f}%)")
print(f"  Frequent itemsets     : {len(frequent_itemsets):,}")
print(f"  Association rules     : {len(rules):,}")
print(f"  Strong rules (lift>=1.5, conf>=35%) : {len(strong_rules)}")
print(f"  {'─'*45}")
print(f"  Metric Statistics:    {'Min':>8} {'Mean':>8} {'Max':>8}")
print(f"  {'─'*45}")
for metric in ["support", "confidence", "lift"]:
    print(f"  {metric.capitalize():<20} "
          f"{rules[metric].min():>8.4f} "
          f"{rules[metric].mean():>8.4f} "
          f"{rules[metric].max():>8.4f}")
print(f"  {'─'*45}")
print(f"  Top rule by LIFT:")
ant = " + ".join(sorted(rules.iloc[0]["antecedents"]))
con = " + ".join(sorted(rules.iloc[0]["consequents"]))
print(f"  {ant}  =>  {con}")
print(f"  Support={rules.iloc[0]['support']:.4f}  "
      f"Confidence={rules.iloc[0]['confidence']:.4f}  "
      f"Lift={rules.iloc[0]['lift']:.4f}")
print("=" * 65)
print("\n  Output files:")
print("    market_basket_analysis.png  — Product support, scatter plots, top rules")
print("    itemset_heatmap.png         — Itemset size distribution & metrics heatmap")
print("=" * 65)
