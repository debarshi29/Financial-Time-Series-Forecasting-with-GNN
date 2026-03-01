from trainer.trainer import *
from data_loader import *
try:
    from model.Thgnn_flexible import *
    print("Using Flexible THGNN Model")
except ImportError:
    from model.Thgnn import *
    print("Using Standard THGNN Model")

import warnings
import torch
import os
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PLOT_DIR = DATA_DIR / "plots"
NIFTY_CSV_DIR = DATA_DIR / "nifty50_csv"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VALIDATION_SPAN = 4

class Args:
    def __init__(self, data_start, data_middle, data_end, pre_data, gpu=0, subtask="regression"):
        self.gpu = str(gpu)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # data settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.pos_adj_dir = "pos_adj_" + self.adj_str
        self.neg_adj_dir = "neg_adj_" + self.adj_str
        self.feat_dir = "features"
        self.label_dir = "label"
        self.mask_dir = "mask"
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        
        # Must match the checkpoint architecture (new_main.py training config)
        self.hidden_dim = 128
        self.num_layers = 1
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        self.target_dim = 1
        self.predictor_activation = None
        
        self.save_path = str(MODEL_DIR)
        self.load_path = self.save_path
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"
        self.predictor_activation = None

    def regression_binary(self):
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"
        self.predictor_activation = None
    
    def classification_binary(self):
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"
        self.predictor_activation = 'sigmoid'

    def classification_tertiary(self):
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"
        self.predictor_activation = 'sigmoid'

def find_top_5_stocks_and_plot(
    val_start_date="2015-01-01", 
    val_end_date="2015-12-31" 
):
    # Expanded range to find files
    print("--- Initializing Data Search ---")
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_DIR}.")

    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]
    date_to_index = {dt: idx for idx, dt in enumerate(file_dates)}

    def _index_for_date(date_str, label):
        dt = pd.to_datetime(date_str).normalize()
        if dt in date_to_index:
            return date_to_index[dt]
        for idx, file_dt in enumerate(file_dates):
            if file_dt >= dt:
                return idx
        return len(file_dates) - 1

    val_start_idx = _index_for_date(val_start_date, "val_start_date")
    val_end_idx = _index_for_date(val_end_date, "val_end_date") + 1
    
    if val_end_idx > len(data_files): val_end_idx = len(data_files)
    
    # Check if range is valid
    if val_start_idx >= val_end_idx:
        print("Warning: Date range invalid or no data found. Using full range.")
        val_start_idx = 0
        val_end_idx = len(data_files)

    pre_data_value = Path(data_files[val_start_idx - 1 if val_start_idx > 0 else 0]).stem
    
    print(f"Validation Range: {file_dates[val_start_idx].strftime('%Y-%m-%d')} to {file_dates[val_end_idx-1].strftime('%Y-%m-%d')}")
    print(f"Using Model Checkpoint Pre-date: {pre_data_value}")

    # Setup Args
    args = Args(
        data_start=0,
        data_middle=val_start_idx,
        data_end=val_end_idx,
        pre_data=pre_data_value,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load Data
    val_dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), mode="val", data_start=0, data_middle=val_start_idx, data_end=val_end_idx)

    # Detect dimension
    _, _, _, sample_labels, _ = extract_data(val_dataset[0], args.device)
    inferred_dim = sample_labels.shape[-1] if sample_labels.dim() > 1 else 1
    args.target_dim = int(inferred_dim)

    # Initialize Model
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                                  out_features=args.out_features, predictor_out_dim=args.target_dim,
                                  predictor_activation=args.predictor_activation).to(args.device)

    # Try to load checkpoint
    # Look for ANY valid checkpoint if pre_data_value specific one is missing
    checkpoint_path = Path(args.load_path) / f"{pre_data_value}_epoch_60.dat"
    
    # Heuristic: Find the LATEST checkpoint file if specific one doesn't exist
    if not checkpoint_path.exists():
        print(f"Exact checkpoint {checkpoint_path} not found. Searching for most recent...")
        available_models = sorted(Path(args.load_path).glob("*_epoch_*.dat"), key=os.path.getmtime)
        if not available_models:
             # Try broader search
             available_models = sorted(Path(args.load_path).glob("*.dat"), key=os.path.getmtime)
        
        if not available_models:
             raise FileNotFoundError("No models found in model_saved directory.")
        
        checkpoint_path = available_models[-1]
        print(f"Using alternative checkpoint: {checkpoint_path}")

    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint) # In case saved directly
        
    model.eval()

    # Collect Results
    data_code = sorted([p.name for p in DAILY_STOCK_DIR.glob("*.csv")])
    # Ensure data_code list aligns with val_dataset
    # The dataset loads files based on sorted listdir of TRAIN_DATA_DIR.
    # daily_stock_dir also has files with dates.
    # We should match by date.
    
    results = []
    
    print("Running Predictions...")
    for i in tqdm(range(len(val_dataset))):
        # We need to know which date this sample corresponds to
        # val_dataset.gnames_all[i] is the filename "YYYY-MM-DD.pkl"
        pkl_name = val_dataset.gnames_all[i]
        date_str = Path(pkl_name).stem
        daily_file = DAILY_STOCK_DIR / f"{date_str}.csv"
        
        if not daily_file.exists():
            continue
            
        df = pd.read_csv(daily_file, dtype=object)
        
        # Get data
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj)
        
        preds = logits.detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()
        
        if preds.ndim == 1: preds = preds[:, None]
        if lbls.ndim == 1: lbls = lbls[:, None]
        
        # Assuming we care about the first target dimension
        pred_vals = preds[:, 0]
        actual_vals = lbls[:, 0]
        
        # Mapping
        for idx, row in df.iterrows():
            if idx < len(pred_vals):
                results.append({
                    'dt': pd.to_datetime(row['dt']),
                    'code': row['code'],
                    'actual_return': float(actual_vals[idx]),
                    'predicted_return': float(pred_vals[idx])
                })

    df_results = pd.DataFrame(results)
    
    # Calculate Error Metrics per Stock
    print("Calculating Metrics...")
    metrics = []
    grouped = df_results.groupby('code')
    
    for code, group in grouped:
        if len(group) < 10: continue 
        mse = np.mean((group['actual_return'] - group['predicted_return'])**2)
        metrics.append({'code': code, 'mse': mse})
        
    df_metrics = pd.DataFrame(metrics)
    if len(df_metrics) == 0:
        print("No valid metrics computed")
        return

    # Find Top 5 Best Stocks (Lowest MSE)
    top_5 = df_metrics.sort_values('mse').head(5)
    print(f"\nTop 5 Best Stocks:\n{top_5}")
    
    for _, row in top_5.iterrows():
        code = row['code']
        mse = row['mse']
        print(f"Processing {code} (MSE: {mse:.6f})...")
        
        stock_data = df_results[df_results['code'] == code].sort_values('dt')
        if stock_data.empty:
            continue

        # Per-stock diagnostics for return prediction quality.
        pred_ret = stock_data['predicted_return'].astype(float).values
        act_ret = stock_data['actual_return'].astype(float).values
        if len(pred_ret) < 2:
            ic = np.nan
        else:
            ic = np.corrcoef(pred_ret, act_ret)[0, 1]
        sign_acc = np.mean(np.sign(pred_ret) == np.sign(act_ret))
        pred_std = float(np.std(pred_ret))
        act_std = float(np.std(act_ret))
        spread_ratio = pred_std / act_std if act_std > 0 else np.nan
        mean_pred = float(np.mean(pred_ret))
        mean_act = float(np.mean(act_ret))
        print(
            f"  IC={ic:.4f}, SignAcc={sign_acc:.2%}, "
            f"PredStd={pred_std:.6f}, ActStd={act_std:.6f}, SpreadRatio={spread_ratio:.3f}"
        )
        
        # Load Price Data
        # Map Code to CSV: ITC.NS -> ITC_NS.csv
        csv_name = code.replace('.', '_') + '.csv'
        csv_path = NIFTY_CSV_DIR / csv_name
        
        if not csv_path.exists():
            print(f"Warning: Price file {csv_path} not found for {code}")
            continue
            
        df_price = pd.read_csv(csv_path)
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        
        # Merge
        # merged has [dt, code, actual_return (R_t+1), predicted_return (Pred_R_t+1), Close (P_t)]
        merged = pd.merge(stock_data, df_price, left_on='dt', right_on='Date', how='inner')
        if len(merged) == 0:
            print(f"No price overlap for {code}")
            continue
            
        # Reconstruct Price Logic
        # The 'label' in the dataset corresponds to the return of the NEXT day (t+1).
        # The 'features' at 'dt' correspond to day 't'.
        # 'Close' in df_price at 'dt' is P_t.
        # So Actual Next Price P_{t+1} = P_t * (1 + actual_return)
        # Predicted Next Price Pred_P_{t+1} = P_t * (1 + predicted_return)
        
        merged['actual_next_price'] = merged['Close'] * (1 + merged['actual_return'])
        merged['predicted_next_price'] = merged['Close'] * (1 + merged['predicted_return'])
        
        # For verification, we can also get the REAL P_{t+1} from the CSV by shifting
        # But (1+R) logic is robust if splits happen, assuming R accounts for it.
        # Let's stick to the return-derived price to match the training objective.
        
        # Plotting
        # We plot "Next Day Price" against the date "t".
        # So at x=Jan 1st, we show the price forecast for Jan 2nd.
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Returns (Next Day Return)
        axes[0].plot(merged['dt'], merged['actual_return'], label='Actual Return (t+1)', color='black', alpha=0.5)
        axes[0].plot(merged['dt'], merged['predicted_return'], label='Predicted Return (t+1)', color='blue', alpha=0.9)
        axes[0].set_title(f"{code} - Next Day Returns (MSE: {mse:.6f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylabel("Return")
        metrics_text = (
            f"IC: {ic:.4f}\n"
            f"Sign Acc: {sign_acc:.2%}\n"
            f"Pred Std: {pred_std:.6f}\n"
            f"Act Std: {act_std:.6f}\n"
            f"Spread Ratio: {spread_ratio:.3f}\n"
            f"Mean Pred: {mean_pred:.6f}\n"
            f"Mean Act: {mean_act:.6f}"
        )
        axes[0].text(
            0.01,
            0.98,
            metrics_text,
            transform=axes[0].transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        # Plot 2: Price (Next Day Price)
        axes[1].plot(merged['dt'], merged['actual_next_price'], label='Actual Next Price (P_t+1)', color='green', linewidth=2)
        axes[1].plot(merged['dt'], merged['predicted_next_price'], label='Predicted Next Price (Pred_P_t+1)', color='red', linestyle='--', alpha=0.8)
        axes[1].plot(merged['dt'], merged['Close'], label='Current Price (P_t) [Baseline]', color='grey', linestyle=':', alpha=0.5)
        axes[1].set_title(f"{code} - Next Day Price Prediction")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylabel("Price")
        axes[1].set_xlabel("Date (Prediction Date t)")
        
        plt.tight_layout()
        save_path = PLOT_DIR / f"top5_{code}.png"
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    # Ensure date range covers available data NOT used in training (training ended before 2024-01-01)
    find_top_5_stocks_and_plot(val_start_date="2024-01-01", val_end_date="2025-12-31")
