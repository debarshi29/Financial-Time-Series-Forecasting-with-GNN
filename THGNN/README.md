# Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction (THGNN)
## 1. Prepare your training data
The repository now includes a full preprocessing pipeline for the Nifty 50 index. The input to the model is a pickle file that contains the stock symbol (`code`), the trading day (`dt`) and six engineered features. Use the helper scripts below to build the dataset and graph structures for the Indian market:

1. **Download and preprocess the market data** (defaults to the Nifty 50 constituents):
   ```bash
   python utils/download_market_data.py --start 2020-01-01 --end 2024-01-01
   python utils/download_market_data.py --start 2020-01-01 --end 2025-11-04 --csv-dir data/nifty50_csv

   > Tip: if Yahoo Finance temporarily blocks requests, rerun the command with
   > `--pause 3 --max-retries 5` to back off between attempts.

   Add `--csv-dir data/nifty50_csv` to also store per-ticker adjusted OHLCV files with
   RSI and MACD indicators (useful for manual inspection or standalone analysis).
   ```
   The command stores the processed dataframe at `data/nifty50.pkl`. You can supply a custom comma-separated ticker list via `--tickers`.

2. **Generate the monthly relation graphs** using sliding-window correlations:
   ```bash
   python utils/generate_relation.py --data-path data/nifty50.pkl --relation-dir data/relation --window 20
   ```
   The script automatically picks the last trading day of each month (within the provided range) and writes the corresponding relation matrices to `data/relation`.

3. **Build the final model inputs** (node features, adjacency matrices, labels and masks):
   ```bash
   python utils/generate_data.py \
       --data-path data/nifty50.pkl \
       --relation-dir data/relation \
       --output-dir data/data_train_predict \
       --daily-stock-dir data/daily_stock
   ```

You can adjust the feature engineering or the graph-building thresholds through the command-line arguments exposed by each utility. The `relation` directory stores the relations between stocks. The `daily_stock` directory contains the stocks that are trained each day. The `data_train_predict` directory stores the final inputs fed to the model each day. The `prediction` directory stores the prediction result of the validation set. The `model_saved` directory stores the trained model.

## ## 2. Train your model

### Configure the trainer

The `fun_train_predict` helper now lets you define the training and validation
splits either by index (the legacy behaviour) or directly by trading date. This
makes it easy to extend the dataset without rewriting code. For example, if you
downloaded and preprocessed data up to **23 October 2025** and want to train on
everything up to that date, rerun the preprocessing utilities with the new
`--end-date` and then call:

```python
from main import fun_train_predict

fun_train_predict(
    data_start=0,                # defaults to the first available sample
    data_middle=None,            # will be overridden by the date arguments
    data_end=None,               # ditto – use all available samples
    train_end_date="2025-10-23",
)
```

You can also pin the validation span via `val_start_date` / `val_end_date` if
you have future data to evaluate on. When dates are not provided the function
falls back to index-based slicing so existing scripts continue to work.

### Install required packages

  ``` shell
  pip install -r requirements.txt  for specific versions
  ```

### Start training

  ``` shell
  sh train.sh
  ```
  ```
## 3. Citing

* If you find **THGNN** is useful for your research, please consider citing the following papers:

  ``` latex
  @inproceedings{Xiang2022Temporal,
    author = {Xiang, Sheng and Cheng, Dawei and Shang, Chencheng and Zhang, Ying and Liang, Yuqi},
    title = {Temporal and Heterogeneous Graph Neural Network for Financial Time Series Prediction},
    year = {2022},
    isbn = {9781450392365},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3511808.3557089},
    doi = {10.1145/3511808.3557089},
    booktitle = {Proceedings of the 31st ACM International Conference on Information & Knowledge Management},
    pages = {3584–3593},
    numpages = {10},
    location = {Atlanta, GA, USA},
    series = {CIKM '22}
}
  ```