# Data Leakage Analysis Report - THGNN

## Executive Summary

**CRITICAL DATA LEAKAGE DETECTED**: The model achieves suspiciously low MSE because it has access to **future correlation information** when making predictions. The adjacency (relation) matrix for each day is calculated using data from that entire month, including days that haven't occurred yet relative to the prediction date.

---

## Leakage Source: Future Correlation Information

### The Problem

In `utils/generate_data.py`, the graph generation process works as follows:

1. **Relation files are generated monthly** (e.g., `2020-01-31.csv`)
2. **Each relation file** uses a 20-day window ending on that month-end date
3. **All days in that month** use the SAME relation matrix from month-end

### Example of Leakage

| Date | What the Model Sees | The Problem |
|------|---------------------|-------------|
| Jan 2, 2020 | Relation matrix from Jan 31, 2020 | This relation was calculated using data from Jan 3-31, which is **future** relative to Jan 2 |
| Jan 15, 2020 | Relation matrix from Jan 31, 2020 | This relation includes correlations from Jan 15-31, which is **future** relative to Jan 15 |

### Code Evidence

From `utils/generate_data.py` (lines 140-155):

```python
for relation_path in relation_files:
    relation_dt = pd.to_datetime(relation_path.stem)  # e.g., 2020-01-31
    ...
    month_start = relation_dt.replace(day=1)  # 2020-01-01
    month_days = [d for d in dates if month_start <= d <= relation_dt]  # ALL days in Jan
    for end_data in tqdm(month_days):  # Jan 2, Jan 3, ..., Jan 31
        ...
        result = {
            "pos_adj": Variable(pos_adj),  # SAME matrix for ALL days in Jan!
            "neg_adj": Variable(neg_adj),
            ...
        }
```

The `pos_adj` and `neg_adj` are the SAME for all days in January, calculated from the full month's data.

---

## Impact of the Leakage

### Why MSE is Near Zero

The Graph Attention Network (GAT) uses the adjacency matrix to aggregate information from correlated stocks. When the adjacency matrix contains future correlation information:

1. **The model knows which stocks will move together in the future**
2. **It can propagate return information through the graph edges**
3. **This makes prediction almost trivial**, resulting in extremely low MSE

### Quantitative Evidence

```
Graph data for 2020-01-02:
- Pos edges: 1562 (matches Jan 2020 relation)
- Neg edges: 136

Expected if no leakage:
- Should use Dec 2019 relation: 1360 edges

Actual behavior:
- Uses Jan 2020 relation: 1562 edges (+15% more edges!)
- These extra edges are based on future data
```

---

## Secondary Leakage Sources

### 1. Train/Validation Split May Have Overlap

In `data_loader.py`, the data split is by index:

```python
if mode == "train":
    self.gnames_all = self.gnames_all[self.data_start:self.data_middle]
elif mode == "val":
    self.gnames_all = self.gnames_all[self.data_middle:self.data_end]
```

If `data_middle` is not carefully chosen, there could be:
- **Temporal overlap**: Training data from future dates leaking into validation
- **Stock overlap**: Same stocks appearing in both sets with adjacent time windows

### 2. Feature Calculation May Include Future Information

The features (open, high, low, close, to, vol) are percentage changes, which should be safe. However:
- **Volume outliers**: Some volume values show >1000% changes (e.g., 41.11 = 4111%)
- These extreme values may carry predictive signal that shouldn't exist

---

## Recommended Fixes

### Fix 1: Use Rolling Window for Relations (CRITICAL)

**Current (WRONG)**:
```python
# One relation matrix per month, used for ALL days in that month
relation_dt = pd.to_datetime(relation_path.stem)  # 2020-01-31
month_days = [d for d in dates if month_start <= d <= relation_dt]
```

**Fixed (CORRECT)**:
```python
# Each day should use a relation matrix calculated from the 20 days BEFORE it
for end_data in trading_days:
    # Calculate relation using window [end_data - 20, end_data]
    # This ensures no future data is used
    relation_window = df[(df["dt"] >= end_data - 20 days) & (df["dt"] <= end_data)]
    pos_adj, neg_adj = calculate_relation(relation_window)
```

### Fix 2: Generate Daily Relation Files

Instead of monthly relation files, generate them daily:

```python
# generate_relation.py should output:
# data/relation_daily/2020-01-02.csv
# data/relation_daily/2020-01-03.csv
# ...
# Each using the 20-day window ending on that date
```

Then modify `generate_data.py` to look up the appropriate daily relation file.

### Fix 3: Purge Future Information

Ensure the relation calculation window ends on or before the prediction date:

```python
# For graph dated 2020-01-02:
relation_end_date = "2020-01-02"  # NOT "2020-01-31"
relation_window = get_window_ending_on(relation_end_date, window_size=20)
```

---

## Verification Steps

After fixing, verify:

1. **Check relation dates match graph dates**:
   ```bash
   # Graph file 2020-01-02.pkl should use relation 2020-01-02.csv
   # NOT 2020-01-31.csv
   ```

2. **Expect higher MSE**:
   - Realistic MSE for stock returns: ~0.0001 to 0.001
   - Current MSE: ~0.000001 (too low!)

3. **Information Coefficient (IC) should drop**:
   - Current IC is likely inflated due to leakage
   - Realistic daily IC: 0.02-0.05 for good models

---

## Additional Checks

Run this to verify the leakage exists in your data:

```python
import pickle
import pandas as pd

# Check a graph file
with open('data/data_train_predict/2020-01-02.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Graph date: 2020-01-02")
print(f"Pos adj sum: {data['pos_adj'].sum().item():.0f}")

# Compare with relation files
rel_dec = pd.read_csv('data/relation/2019-12-31.csv', index_col=0)
rel_jan = pd.read_csv('data/relation/2020-01-31.csv', index_col=0)

# Build adjacency matrices
import networkx as nx
import numpy as np

def build_adj(df, threshold=0.1):
    G = nx.Graph(df > threshold)
    adj = nx.adjacency_matrix(G).toarray() - np.diag(np.diag(nx.adjacency_matrix(G).toarray()))
    return adj.sum()

print(f"Dec 2019 relation edges: {build_adj(rel_dec)}")
print(f"Jan 2020 relation edges: {build_adj(rel_jan)}")

# If Jan 2 graph edges match Jan 2020 relation, leakage confirmed!
```

---

## Conclusion

The THGNN model as currently implemented has **critical data leakage** through the relation/adjacency matrix. The model sees future correlation patterns when making predictions, which explains the suspiciously low MSE scores.

**Immediate action required**: Fix the relation generation to use rolling windows that do not include future data.

**Estimated impact**: After fixing, expect:
- MSE to increase by 10-100x
- IC to drop to more realistic levels
- Strategy returns to be more modest
