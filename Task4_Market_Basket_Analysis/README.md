# Task 4: Retail – Market Basket Analysis (Association Rules)

**Growfinix Machine Learning Internship | Task 4 | Domain: Retail**

---

## Objective

Discover **frequent product combinations** and **purchasing patterns** in supermarket transaction data using the **Apriori algorithm**. Association rules reveal which products customers tend to buy together, enabling smarter product placement, promotions, and recommendations.

---

## Dataset

A synthetic supermarket transaction dataset is generated directly by the script — no external download required. It contains **2,000 customer receipts** across **30 common grocery products**.

### Products Covered

| Category | Products |
|---|---|
| **Dairy** | Milk, Eggs, Butter, Cheese, Yogurt |
| **Proteins** | Chicken, Fish |
| **Staples** | Bread, Rice, Pasta |
| **Vegetables** | Tomatoes, Onions, Potatoes, Lettuce, Carrots |
| **Fruits** | Apples, Bananas, Oranges, Grapes, Strawberries |
| **Beverages** | Coffee, Tea, Juice, Soda, Water |
| **Snacks/Other** | Chips, Cookies, Chocolate, Cake, Soap |

### Dataset Design

- **2,000 transactions** with realistic, probabilistic item selection
- Popular items (Bread, Milk, Eggs, Bananas) have higher base purchase probabilities
- Each basket averages ~7 items

---

## Steps Performed

| Step | Description |
|---|---|
| 1 | Generate 2,000-transaction synthetic supermarket dataset |
| 2 | Inspect shape, dtypes, basket size statistics |
| 3 | Remove duplicate transactions; handle missing values |
| 4 | Parse item lists; compute per-product purchase counts |
| 5 | Encode transactions using `TransactionEncoder` (binary one-hot matrix) |
| 6 | Run **Apriori** with `min_support=0.05` to find frequent itemsets |
| 7 | Generate association rules with `metric=lift, min_threshold=1.0` |
| 8 | Display **support, confidence, and lift** for all rules |
| 9 | Identify strong product relationships (lift ≥ 1.5, confidence ≥ 35%) |
| 10 | Visualise top rules, scatter plots, and metrics heatmap |

---

## Key Concepts

### Association Rule: `A => B`

| Metric | Formula | Meaning |
|---|---|---|
| **Support** | `P(A ∩ B)` | Fraction of transactions containing both A and B |
| **Confidence** | `P(B \| A)` | How often B is bought when A is bought |
| **Lift** | `Conf / P(B)` | How much more likely B is given A vs random chance |

> **Lift > 1** → positive association (buying A increases the chance of buying B)  
> **Lift = 1** → no association (independent)  
> **Lift < 1** → negative association

### Apriori Algorithm

1. Find all itemsets that exceed `min_support`
2. Prune: if a subset is infrequent, all its supersets are also infrequent
3. Iteratively generate 1-item → 2-item → 3-item → ... frequent itemsets

---

## Parameters Used

| Parameter | Value | Meaning |
|---|---|---|
| `min_support` | 0.05 | Itemset must appear in ≥ 5% of transactions |
| `min_confidence` | 0.20 | Rule must have ≥ 20% confidence |
| `min_lift` | 1.0 | Only rules with positive association |
| `max_len` | 4 | Itemsets up to 4 products |

---

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib seaborn mlxtend
```

### Run

```bash
python market_basket_analysis.py
```

> **No dataset download required.** The script generates data automatically.

---

## Output Files

| File | Description |
|---|---|
| `market_basket_analysis.png` | Product support bar chart, support vs confidence scatter, support vs lift scatter, top-15 rules by lift and confidence |
| `itemset_heatmap.png` | Frequent itemset size distribution + metrics heatmap for top-20 rules |

---

## Interpreting Results

- **High lift rules** → strong product affinity; ideal for cross-promotions and bundling
- **High confidence rules** → reliable predictions; ideal for recommendation engines ("customers who bought X also bought Y")
- **High support rules** → very common combinations; ideal for product placement

### Example Retail Actions from Association Rules

| Rule Insight | Retail Action |
|---|---|
| Bread → Butter (high confidence) | Place Butter near Bread aisle |
| Milk + Eggs → Cheese (high lift) | Offer combo discount |
| Coffee → Cookies (moderate lift) | Suggest at checkout |

---

## Key Design Decisions

- **Synthetic dataset** — realistic purchase probability distribution; popular items weighted higher; self-contained with no API dependency
- **`TransactionEncoder`** — mlxtend's built-in encoder creates a clean boolean DataFrame from raw transaction lists
- **`apriori()` with `use_colnames=True`** — returns frozensets of product names (readable) instead of column indices
- **`association_rules()` with `metric='lift'`** — sorted by lift first; confidence filter applied post-generation to retain all rules and let the analyst choose thresholds
- **`max_len=4`** — caps itemset length to prevent combinatorial explosion while still capturing multi-product patterns

---

## Project Structure

```
Task4_Market_Basket_Analysis/
├── market_basket_analysis.py   ← Main Python script
└── README.md                   ← This file
```
