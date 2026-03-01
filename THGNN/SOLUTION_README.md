# GRU Shape Mismatch Solution

## Problem Diagnosis

You're experiencing a GRU shape mismatch when changing `data_start`, `data_middle`, and `data_end` values with your 10-year dataset (2672 files from 2015-2025).

### Root Cause

The GRU in `model/Thgnn.py` expects:
- **Input size**: 6 (features: open, high, low, close, to, vol)
- **Sequence length**: Determined by the `--window` parameter used during data generation (default: 20)

The shape mismatch occurs when:
1. Different data files have different temporal window sizes
2. The model is hardcoded to expect a specific sequence length
3. Your new 10-year dataset may have been generated with a different window size than expected

### Current Data Range

With `data_start=20, data_middle=39, data_end=43`:
- **Training**: Files 20-38 (2015-03-02 to 2015-03-27) = 19 files
- **Validation**: Files 39-42 (2015-03-30 to 2015-04-06) = 4 files

## Solutions

### Solution 1: Check Data Generation Parameters (Recommended)

1. **Verify the window size** used when generating your 10-year dataset:
   ```bash
   # Check if you have a data generation log or script
   # The window parameter should be consistent across all data
   ```

2. **Regenerate data with consistent parameters** if needed:
   ```bash
   python utils/generate_data.py --window 20 --horizon 3 --start-date 2015-01-01 --end-date 2025-12-31
   ```

### Solution 2: Make Model Flexible (Quick Fix)

Modify the model to automatically detect the sequence length from the input data.

See `model/Thgnn_flexible.py` for the updated implementation.

### Solution 3: Adjust Data Indices

If you have a 10-year dataset but want to use specific years:

```python
# For example, to use 2020-2021 data:
# First, find the indices corresponding to those dates
data_files = sorted([p.name for p in Path("data/data_train_predict").glob("*.pkl")])

# Find indices for your desired date range
start_idx = next(i for i, f in enumerate(data_files) if f >= "2020-01-01.pkl")
middle_idx = next(i for i, f in enumerate(data_files) if f >= "2020-06-01.pkl")
end_idx = next(i for i, f in enumerate(data_files) if f >= "2021-01-01.pkl")

# Use these indices in main.py
data_start = start_idx
data_middle = middle_idx
data_end = end_idx
```

## Implementation

I've created a flexible version of the model that will be saved as `model/Thgnn_flexible.py`.

### Key Changes:
1. Removed hardcoded `in_features=6` from GRU initialization
2. Added dynamic feature detection in the forward pass
3. Model now adapts to any input shape automatically

### How to Use:

1. **Option A**: Replace the existing model
   ```bash
   # Backup original
   cp model/Thgnn.py model/Thgnn_original.py
   # Use flexible version
   cp model/Thgnn_flexible.py model/Thgnn.py
   ```

2. **Option B**: Update main.py to use the flexible model
   ```python
   # In main.py, change the import
   from model.Thgnn_flexible import *
   ```

## Verification Steps

After applying the fix:

1. **Run the diagnostic script**:
   ```bash
   python diagnose_gru_issue.py
   ```

2. **Test with your current settings**:
   ```bash
   python main.py
   ```

3. **Monitor the output** for any shape-related errors

## Additional Notes

- The original model expected exactly 6 features and 20 time steps
- Your 10-year dataset should be consistent, but the model should be flexible
- If you still get errors, check that all pickle files have the same feature structure
- Consider using a smaller date range for initial testing (e.g., data_start=20, data_middle=30, data_end=35)
