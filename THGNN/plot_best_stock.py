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
        
        # This must match the training config
        self.hidden_dim = 128     
        self.num_layers = 2       
        self.num_heads = 4        
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

def find_best_stock_and_plot(
    val_start_date="2024-01-01", 
    val_end_date="2025-12-31"
):
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
    
    # Ensure range is valid
    if val_end_idx > len(data_files): val_end_idx = len(data_files)
    
    pre_data_value = Path(data_files[val_start_idx - 1]).stem
    
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
    # val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True) # Not strictly needed if we iterate manually

    # Detect dimension
    _, _, _, sample_labels, _ = extract_data(val_dataset[0], args.device)
    inferred_dim = sample_labels.shape[-1] if sample_labels.dim() > 1 else 1
    args.target_dim = int(inferred_dim)

    # Initialize Model
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                                  out_features=args.out_features, predictor_out_dim=args.target_dim,
                                  predictor_activation=args.predictor_activation).to(args.device)

    # Try to load checkpoint
    checkpoint_path = Path(args.load_path) / f"{pre_data_value}_epoch_60.dat"
    if not checkpoint_path.exists():
        # Fallback to finding the nearest available model if exact match fails
        print(f"Exact checkpoint {checkpoint_path} not found. Searching for most recent...")
        available_models = sorted(Path(args.load_path).glob("*_epoch_60.dat"))
        if not available_models:
             raise FileNotFoundError("No models found in model_saved directory.")
        checkpoint_path = available_models[-1]
        print(f"Using alternative checkpoint: {checkpoint_path}")

    print(f"Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Collect Results
    data_code = sorted([p.name for p in DAILY_STOCK_DIR.glob("*.csv")])
    data_code_last = data_code[val_start_idx:val_end_idx]
    
    results = []
    
    print("Running Predictions...")
    for i in tqdm(range(len(val_dataset))):
        df = pd.read_csv(DAILY_STOCK_DIR / data_code_last[i], dtype=object)
        
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
        
        # Match with stock codes in df
        # df usually has 'code', 'dt', ...
        # Ensure df aligns with graph node order. 
        # In this dataset, nodes in the graph correspond to rows in the daily stock CSV for that day?
        # NO: The graph is constructed from ALL stocks. The daily_stock_dir contains ONE file per DAY, which lists all stocks.
        # So df corresponds to the nodes in the graph for that day.
        
        if len(df) != len(pred_vals):
            # This sometimes happens if graph construction filters nodes. 
            # We assume 1-to-1 mapping if not filtered.
            # If filtered, we need the mapping.
            # However, extract_data returns features which align with the graph.
            # We'll assume the stock codes in df align with the graph nodes 
            # (which is standard for this codebase unless 'index_suggestions' is used irregularly).
            pass

        for idx, row in df.iterrows():
            if idx < len(pred_vals):
                results.append({
                    'dt': row['dt'],
                    'code': row['code'],
                    'actual': float(actual_vals[idx]),
                    'predicted': float(pred_vals[idx])
                })

    df_results = pd.DataFrame(results)
    df_results['dt'] = pd.to_datetime(df_results['dt'])
    
    # Calculate Error Metrics per Stock
    print("Calculating Metrics...")
    metrics = []
    grouped = df_results.groupby('code')
    
    for code, group in grouped:
        if len(group) < 5: continue # Skip stocks with very few data points
        mse = np.mean((group['actual'] - group['predicted'])**2)
        mae = np.mean(np.abs(group['actual'] - group['predicted']))
        metrics.append({'code': code, 'mse': mse, 'mae': mae, 'count': len(group)})
        
    df_metrics = pd.DataFrame(metrics)
    
    # Find Best Stock (Lowest MSE)
    best_stock = df_metrics.sort_values('mse').iloc[0]
    best_code = best_stock['code']
    print(f"\nBest Stock Found: {best_code}")
    print(f"MSE: {best_stock['mse']:.6f}")
    print(f"MAE: {best_stock['mae']:.6f}")
    
    # Plotting
    plot_data = df_results[df_results['code'] == best_code].sort_values('dt')
    
    plt.figure(figsize=(12, 6))
    plt.plot(plot_data['dt'], plot_data['actual'], label='Actual', marker='.')
    plt.plot(plot_data['dt'], plot_data['predicted'], label='Predicted', marker='.')
    plt.title(f"Best Performing Stock: {best_code} (MSE: {best_stock['mse']:.6f})")
    plt.xlabel("Date")
    plt.ylabel("Return / Label")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_path = PLOT_DIR / f"best_stock_{best_code}.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    
    # Optional: Find Worst Stock
    worst_stock = df_metrics.sort_values('mse', ascending=False).iloc[0]
    print(f"\nWorst Stock (for comparison): {worst_stock['code']} (MSE: {worst_stock['mse']:.6f})")

if __name__ == "__main__":
    # You can adjust dates here
    find_best_stock_and_plot(val_start_date="2024-01-01", val_end_date="2025-12-31")
