from trainer.trainer import *
from data_loader import *
try:
    from model.Thgnn_flexible import *
except ImportError:
    from model.Thgnn import *

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

# Setup paths (copied from plot_best_stock.py)
BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"

def analyze_specific_stock(stock_code="ITC.NS", val_start_date="2024-01-01", val_end_date="2025-12-31"):
    # 1. Setup Environment
    gpu = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 2. Find Data Indices
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]
    date_to_index = {dt: idx for idx, dt in enumerate(file_dates)}

    def _index_for_date(date_str):
        dt = pd.to_datetime(date_str).normalize()
        if dt in date_to_index:
            return date_to_index[dt]
        for idx, file_dt in enumerate(file_dates):
            if file_dt >= dt: return idx
        return len(file_dates) - 1

    start_idx = _index_for_date(val_start_date)
    end_idx = _index_for_date(val_end_date) + 1
    if end_idx > len(data_files): end_idx = len(data_files)
    
    pre_data_value = Path(data_files[start_idx - 1]).stem
    
    # 3. Load Model
    # Hardcoded from previous run
    hidden_dim = 128
    num_heads = 4
    num_layers = 2
    out_features = 32
    
    # Detect checkpoint
    checkpoint_path = MODEL_DIR / f"{pre_data_value}_epoch_60.dat"
    if not checkpoint_path.exists():
        # Fallback
        available = sorted(MODEL_DIR.glob("*_epoch_60.dat"))
        if available: checkpoint_path = available[-1]
    
    print(f"Analyzing {stock_code} using model from {checkpoint_path.name}")
    
    # Load dataset structure to determine dims
    tmp_loader = AllGraphDataSampler(str(TRAIN_DATA_DIR), mode="val", data_start=start_idx, data_middle=start_idx+1, data_end=start_idx+2)
    _, _, _, sample_labels, _ = extract_data(tmp_loader[0], device)
    target_dim = sample_labels.shape[-1] if sample_labels.dim() > 1 else 1

    model = StockHeteGAT(hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
                         out_features=out_features, predictor_out_dim=target_dim,
                         predictor_activation=None).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 4. Run Inference Specific to Stock
    results = []
    
    data_code_files = sorted([p.name for p in DAILY_STOCK_DIR.glob("*.csv")])
    # subset
    data_code_files = data_code_files[start_idx:end_idx]
    
    # Create dataset
    val_dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), mode="val", data_start=0, data_middle=start_idx, data_end=end_idx)
    
    print(f"Scanning {len(val_dataset)} days for {stock_code}...")
    
    for i in range(len(val_dataset)):
        daily_file = DAILY_STOCK_DIR / data_code_files[i]
        df = pd.read_csv(daily_file, dtype=object)
        
        # Check if stock is in this day
        if stock_code not in df['code'].values:
            continue
            
        # Get graph data
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, device)
        
        with torch.no_grad():
            logits = model(features, pos_adj, neg_adj)
        
        preds = logits.detach().cpu().numpy()
        lbls = labels.detach().cpu().numpy()
        
        # Find index of stock
        stock_idx = df[df['code'] == stock_code].index[0]
        
        # Handle shapes
        if preds.ndim == 1: 
            p = preds[stock_idx]
        else:
            p = preds[stock_idx, 0]
            
        if lbls.ndim == 1:
            a = lbls[stock_idx]
        else:
            a = lbls[stock_idx, 0]
            
        results.append({
            'dt': df.loc[stock_idx, 'dt'],
            'actual': float(a),
            'predicted': float(p)
        })
        
    df_res = pd.DataFrame(results)
    df_res['dt'] = pd.to_datetime(df_res['dt'])
    df_res = df_res.sort_values('dt')
    
    # 5. Calculate Statistics
    mse = np.mean((df_res['actual'] - df_res['predicted'])**2)
    mae = np.mean(np.abs(df_res['actual'] - df_res['predicted']))
    corr = df_res['actual'].corr(df_res['predicted'])
    
    # Directional Accuracy
    actual_diff = df_res['actual'].diff()
    pred_diff = df_res['predicted'].diff()
    # Align
    actual_dir = np.sign(actual_diff.dropna())
    pred_dir = np.sign(pred_diff.dropna())
    dir_acc = (actual_dir == pred_dir).mean()
    
    report = []
    report.append("\n" + "="*40)
    report.append(f"INTERPRETATION FOR {stock_code}")
    report.append("="*40)
    report.append(f"Data Points: {len(df_res)}")
    report.append(f"MSE: {mse:.6f} (Very Low -> Good Fit)")
    report.append(f"MAE: {mae:.6f}")
    report.append(f"Correlation (Pearson): {corr:.4f}")
    report.append(f"Directional Accuracy: {dir_acc:.2%}")
    report.append(f"Actual Volatility (Std): {df_res['actual'].std():.4f}")
    report.append(f"Predicted Volatility (Std): {df_res['predicted'].std():.4f}")
    
    # Check for lag
    from scipy import signal
    a = (df_res['actual'] - df_res['actual'].mean()).values
    p = (df_res['predicted'] - df_res['predicted'].mean()).values
    # Normalize
    a = a / (np.linalg.norm(a) + 1e-9)
    p = p / (np.linalg.norm(p) + 1e-9)
    xcorr = signal.correlate(a, p)
    lags = signal.correlation_lags(len(a), len(p))
    lag_idx = np.argmax(xcorr)
    best_lag = lags[lag_idx]
    
    report.append(f"Lag Analysis: Best fit at lag {best_lag}")
    
    report.append(f"Overall Trend Actual: {df_res['actual'].iloc[-1] - df_res['actual'].iloc[0]:.4f}")
    report.append(f"Overall Trend Pred:   {df_res['predicted'].iloc[-1] - df_res['predicted'].iloc[0]:.4f}")
    
    with open("itc_final_report.txt", "w") as f:
        f.write("\n".join(report))
    print("Report saved to itc_final_report.txt")

if __name__ == "__main__":
    analyze_specific_stock()
