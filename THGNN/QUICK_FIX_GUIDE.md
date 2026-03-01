# GRU Shape Mismatch - Quick Fix Guide

## Problem Summary

You're getting a GRU shape mismatch error when using your 10-year dataset (2672 files from 2015-2025) with the indices:
- `data_start = 20`
- `data_middle = 39` 
- `data_end = 43`

## Root Cause

The GRU model expects input with shape `(num_stocks, window_size, 6)` where:
- `window_size` is the temporal sequence length (typically 20 days)
- `6` is the number of features (open, high, low, close, to, vol)

The error occurs when:
1. Data files have inconsistent window sizes
2. The model's hardcoded `in_features=6` doesn't match your data
3. The dropout parameter causes issues with `num_layers=1`

## What I've Fixed

### 1. Fixed the Dropout Issue ✓
**File**: `model/Thgnn.py`

Changed line 116 from:
```python
dropout=0.1
```
to:
```python
dropout=0.1 if num_layers > 1 else 0.0
```

This prevents PyTorch warnings and potential issues when using a single-layer GRU.

### 2. Created a Flexible Model Version ✓
**File**: `model/Thgnn_flexible.py`

This version automatically detects the input feature dimension and reinitializes the GRU if needed. It's more robust for datasets with varying dimensions.

## How to Fix Your Issue

### Option 1: Use the Fixed Original Model (Recommended for now)

The fix to `model/Thgnn.py` should resolve the dropout warning. However, if you're still getting shape mismatches, it means your data has inconsistent dimensions.

**To verify**, check what window size was used when generating your data:
```bash
# Look for the window parameter in your data generation command
# It should be in your data generation logs or scripts
```

### Option 2: Use the Flexible Model

Replace the import in `main.py`:

```python
# Change this line (around line 1):
from model.Thgnn import *

# To this:
from model.Thgnn_flexible import *
```

The flexible model will automatically adapt to your input dimensions.

### Option 3: Adjust Your Data Indices

Your current indices use very early data (March-April 2015). With a 10-year dataset, you might want to use more recent or more data:

**Current setup**:
- Training: Files 20-38 (19 files, ~1 month of data)
- Validation: Files 39-42 (4 files, ~1 week of data)

**Suggested alternatives**:

1. **Use more training data** (e.g., 6 months):
   ```python
   data_start = 20
   data_middle = 140  # ~6 months of trading days
   data_end = 160     # ~1 month validation
   ```

2. **Use recent data** (2024-2025):
   ```python
   data_start = 2400  # Approximate index for 2024 data
   data_middle = 2500
   data_end = 2520
   ```

3. **Use the helper script** to find exact indices:
   ```bash
   python find_data_indices.py
   ```

## Diagnostic Steps

1. **Check your data dimensions**:
   ```bash
   python diagnose_gru_issue.py
   ```
   (Note: This requires torch to be installed)

2. **Find appropriate indices**:
   ```bash
   python find_data_indices.py
   ```

3. **Verify the window size** used during data generation:
   - Check `utils/generate_data.py` for the `--window` parameter
   - Default is 20, but verify what was actually used

## Quick Test

Try using a larger, more recent data range:

```python
# In main.py, change lines 315-317 to:
data_start = 100
data_middle = 200
data_end = 220

fun_train_predict(data_start, data_middle, data_end)
```

This gives you:
- 100 training samples (~5 months)
- 20 validation samples (~1 month)

## If You Still Get Errors

1. **Check the exact error message** - is it about:
   - Input size mismatch?
   - Sequence length mismatch?
   - Batch size issues?

2. **Verify data consistency**:
   - All pickle files should have the same feature structure
   - The window size should be consistent across all files

3. **Consider regenerating data**:
   ```bash
   python utils/generate_data.py --window 20 --horizon 3 --start-date 2015-01-01 --end-date 2025-12-31
   ```

## Files Created

1. `model/Thgnn.py` - Fixed dropout issue ✓
2. `model/Thgnn_flexible.py` - Flexible version that auto-detects dimensions
3. `diagnose_gru_issue.py` - Diagnostic tool to check data shapes
4. `find_data_indices.py` - Helper to find appropriate indices by date
5. `SOLUTION_README.md` - Detailed explanation of the issue
6. `QUICK_FIX_GUIDE.md` - This file

## Next Steps

1. Try running with the fixed model
2. If still getting errors, use the flexible model
3. Consider using more training data (increase data_middle)
4. Verify your data was generated with consistent parameters

## Need More Help?

Share the exact error message you're getting, including:
- The full traceback
- The shapes mentioned in the error
- Which line is failing

This will help pinpoint the exact issue.
