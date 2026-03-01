from trainer.trainer import *
from data_loader import *
# Try to import flexible model, fall back to standard if not found
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
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PREDICTION_DIR = DATA_DIR / "prediction"
PLOT_DIR = DATA_DIR / "plots"

for directory in [MODEL_DIR, PREDICTION_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_VALIDATION_SPAN = 4

class Args:
    def __init__(self, data_start, data_middle, data_end, pre_data, gpu=0, subtask="regression"):
        # device
        self.gpu = str(gpu)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n" + "="*50)
        print(f"DEVICE STATUS: Using {self.device.upper()}")
        if self.device == 'cuda':
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
            print(f"  Memory Reserved:  {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")
        else:
            print("  WARNING: GPU not available. Training will be slow on CPU.")
        print("="*50 + "\n")
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
        # epoch settings
        self.max_epochs = 60
        self.epochs_eval = 10
        # learning rate settings
        self.lr = 0.0001 # Reduced from 0.0002 for stability
        self.gamma = 0.3
        # model settings
        self.hidden_dim = 128     # Increased from 64
        self.num_layers = 2       # Increased from 1
        self.num_heads = 4        # 4 heads is sufficient for 128 dim
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        self.target_dim = 1
        self.predictor_activation = None
        # save model settings
        self.save_path = str(MODEL_DIR)
        self.load_path = self.save_path
        self.save_name = self.model_name + "_hidden_" + str(self.hidden_dim) + "_head_" + str(self.num_heads) + \
                         "_outfeat_" + str(self.out_features) + "_batchsize_" + str(self.batch_size) + "_adjth_" + \
                         str(self.adj_str)

        self.epochs_save_by = 60
        self.sub_task = subtask
        eval("self.{}".format(self.sub_task))()

    def regression(self):
        self.save_name = self.save_name + "_reg_rank_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_regression"
        self.mask_dir = self.mask_dir + "_regression"
        self.predictor_activation = None

    def regression_binary(self):
        self.save_name = self.save_name + "_reg_binary_"
        self.loss_fcn = mse_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"
        self.predictor_activation = None
    
    def classification_binary(self):
        self.save_name = self.save_name + "_clas_binary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_twoclass"
        self.mask_dir = self.mask_dir + "_twoclass"
        self.predictor_activation = 'sigmoid'

    def classification_tertiary(self):
        self.save_name = self.save_name + "_clas_tertiary_"
        self.loss_fcn = bce_loss
        self.label_dir = self.label_dir + "_threeclass"
        self.mask_dir = self.mask_dir + "_threeclass"
        self.predictor_activation = 'sigmoid'


def run_backtest_analysis(df_results, output_prefix="backtest"):
    """
    Perform backtest analysis on prediction results.
    df_results: DataFrame with 'dt', 'code', 'score', 'label'
    """
    print("\n--- Starting Backtest Analysis ---")
    
    if 'label' not in df_results.columns:
        print("Warning: 'label' column not found in results. Skipping granular backtest metrics.")
        return

    # Convert date
    df_results['dt'] = pd.to_datetime(df_results['dt'])
    
    # 1. IC Analysis (Information Coefficient)
    daily_ic = df_results.groupby('dt').apply(lambda x: x['score'].corr(x['label']))
    mean_ic = daily_ic.mean()
    print(f"Mean IC: {mean_ic:.4f}")
    
    plt.figure(figsize=(10, 6))
    daily_ic.plot(kind='bar', color='skyblue')
    plt.title(f'Daily Information Coefficient (Mean IC: {mean_ic:.4f})')
    plt.axhline(mean_ic, color='r', linestyle='--', label='Mean IC')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{output_prefix}_ic.png")
    plt.close()
    
    # 2. Cumulative Returns (Long-Short Strategy)
    # Strategy: Buy Top 5, Short Bottom 5
    def get_strategy_return(group):
        k = 5
        if len(group) < 2 * k:
            return 0
        sorted_group = group.sort_values('score', ascending=False)
        longs = sorted_group.head(k)['label'].mean()
        shorts = sorted_group.tail(k)['label'].mean()
        return longs - shorts

    daily_returns = df_results.groupby('dt').apply(get_strategy_return)
    cum_returns = daily_returns.cumsum()
    
    plt.figure(figsize=(10, 6))
    cum_returns.plot(color='green')
    plt.title('Cumulative Returns (Long Top 5 - Short Bottom 5)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Label Return')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{output_prefix}_cumulative_returns.png")
    plt.close()
    
    print(f"Backtest plots saved to {PLOT_DIR}")


def fun_train_predict(
    data_start,
    data_middle,
    data_end,
    pre_data=None,
    prediction_horizon=None,
    target_dim=None,
    prediction_filename="pred.csv",
    walk_forward_filename="walk_forward_pred.csv",
    train_start_date=None,
    train_end_date=None,
    val_start_date=None,
    val_end_date=None,
    validation_span=DEFAULT_VALIDATION_SPAN,
):
    # --- Data Indexing Logic (Same as main_corr.py) ---
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_DIR}.")

    total_samples = len(data_files)
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]
    date_to_index = {dt: idx for idx, dt in enumerate(file_dates)}

    def _index_for_date(date_str, label):
        dt = pd.to_datetime(date_str).normalize()
        # Try exact match first
        if dt in date_to_index:
            return date_to_index[dt]
        
        # If not found, find the first date >= requested date
        for idx, file_dt in enumerate(file_dates):
            if file_dt >= dt:
                print(f"Note: {label} '{date_str}' not found. snapping to next available: {file_dt.strftime('%Y-%m-%d')}")
                return idx
        
        # If date is past the end of all files
        available = ", ".join(sorted({d.strftime("%Y-%m-%d") for d in file_dates[:5]})) + "..."
        # Special case: if we are looking for an END date and it's past the last file, return the total count
        if "end_date" in label:
             print(f"Note: {label} '{date_str}' is past the last file ({file_dates[-1].strftime('%Y-%m-%d')}). Using total length.")
             return len(file_dates) - 1 # Return last valid index so +1 logic later works
             
        raise ValueError(f"{label} '{date_str}' is past the end of the dataset. First few dates: {available}")

    # Determine Indices
    if data_start is None: data_start = 0
    train_start_idx = int(data_start)
    
    if validation_span is None:
        requested_val_span = min(DEFAULT_VALIDATION_SPAN, max(1, total_samples - train_start_idx))
    else:
        requested_val_span = int(validation_span)

    if data_middle is None:
        effective_span = min(requested_val_span, max(1, total_samples - train_start_idx))
        val_start_idx = total_samples - effective_span
    else:
        val_start_idx = int(data_middle)

    if val_start_idx <= train_start_idx: val_start_idx = train_start_idx + 1

    if data_end is None: val_end_idx = total_samples
    else: val_end_idx = int(data_end)

    # Date format overrides
    if train_start_date: train_start_idx = _index_for_date(train_start_date, "train_start_date")
    if train_end_date: val_start_idx = _index_for_date(train_end_date, "train_end_date") + 1
    if val_start_date: val_start_idx = _index_for_date(val_start_date, "val_start_date")
    if val_end_date: val_end_idx = _index_for_date(val_end_date, "val_end_date") + 1

    # Describe split
    def _describe_split(label, start_idx, end_idx):
        if end_idx <= start_idx: return
        start_date = file_dates[start_idx].strftime("%Y-%m-%d")
        end_date = file_dates[end_idx - 1].strftime("%Y-%m-%d")
        print(f"{label} split: {start_date} -> {end_date} ({end_idx - start_idx} samples)")

    _describe_split("Training", train_start_idx, val_start_idx)
    _describe_split("Validation", val_start_idx, val_end_idx)

    # Pre-data string determination
    if pre_data is None:
        pre_data_value = Path(data_files[val_start_idx - 1]).stem
    else:
        pre_data_value = pre_data

    # --- Setup Arguments & Datasets ---
    args = Args(
        data_start=train_start_idx,
        data_middle=val_start_idx,
        data_end=val_end_idx,
        pre_data=pre_data_value,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), data_start=train_start_idx, data_middle=val_start_idx, data_end=val_end_idx)
    val_dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), mode="val", data_start=train_start_idx, data_middle=val_start_idx, data_end=val_end_idx)
    
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)

    # Detect dimension
    _, _, _, sample_labels, _ = extract_data(dataset[0], args.device)
    inferred_dim = sample_labels.shape[-1] if sample_labels.dim() > 1 else 1
    
    if target_dim is None: target_dim = inferred_dim
    args.target_dim = int(target_dim)
    args.save_name += f"_tdim_{args.target_dim}"

    # Initialize Model
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                                  out_features=args.out_features, predictor_out_dim=args.target_dim,
                                  predictor_activation=args.predictor_activation).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    
    # --- Training Loop with Plotting ---
    print('Start Training...')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.max_epochs):
        train_loss = train_epoch(epoch, args, model, dataset_loader, optimizer, scheduler, args.loss_fcn)
        # Ensure loss is a float/cpu value, not a cuda tensor
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.detach().cpu().item()
            
        history['train_loss'].append(train_loss)
        
        if epoch % args.epochs_eval == 0 or epoch == args.max_epochs - 1:
            eval_loss, _ = eval_epoch(args, model, val_dataset_loader, args.loss_fcn)
            if isinstance(eval_loss, torch.Tensor):
                eval_loss = eval_loss.detach().cpu().item()
                
            history['val_loss'].append(eval_loss)
            print(f'Epoch: {epoch + 1}/{args.max_epochs}, train loss: {train_loss:.6f}, val loss: {eval_loss:.6f}')
        else:
            print(f'Epoch: {epoch + 1}/{args.max_epochs}, train loss: {train_loss:.6f}')
            # Fill gap for plotting
            if history['val_loss']: history['val_loss'].append(history['val_loss'][-1])
            else: history['val_loss'].append(train_loss)

        if (epoch + 1) % args.epochs_save_by == 0:
            print("Saving model checkpoint...")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, Path(args.save_path) / f"{pre_data_value}_epoch_{epoch + 1}.dat")

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(PLOT_DIR / 'loss_curve.png')
    plt.close()

    # --- Prediction & Backtest ---
    print("\nRunning Evaluation & Backtest...")
    checkpoint_path = Path(args.load_path) / f"{pre_data_value}_epoch_{epoch + 1}.dat"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])

    data_code = sorted([p.name for p in DAILY_STOCK_DIR.glob("*.csv")])
    data_code_last = data_code[val_start_idx:val_end_idx]
    
    df_score = pd.DataFrame()
    full_backtest_data = []

    for i in tqdm(range(len(val_dataset)), desc="Evaluating"):
        df = pd.read_csv(DAILY_STOCK_DIR / data_code_last[i], dtype=object)
        tmp_data = val_dataset[i]
        pos_adj, neg_adj, features, labels, mask = extract_data(tmp_data, args.device)
        
        # Capture labels for backtest
        labels_np = labels.detach().cpu().numpy()
        
        model.eval()
        logits = model(features, pos_adj, neg_adj)
        predictions = logits.detach().cpu().numpy()
        
        if predictions.ndim == 1: predictions = predictions[:, None]
        
        # We need the first column of predictions/labels usually for single-step forecast
        # Append to full backtest data
        # Assuming df has 'code' and 'dt'
        current_dt = df['dt'].iloc[0]
        for idx_row, row in df.iterrows():
            full_backtest_data.append({
                'dt': row['dt'],
                'code': row['code'],
                'score': predictions[idx_row, 0],
                'label': labels_np[idx_row, 0] if labels_np.ndim > 1 else labels_np[idx_row]
            })

        if predictions.shape[1] == 1: score_columns = ["score"]
        else: score_columns = [f"score_t+{idx + 1}" for idx in range(predictions.shape[1])]
        
        scores_df = pd.DataFrame(predictions, columns=score_columns)
        combined_df = pd.concat([df.reset_index(drop=True), scores_df], axis=1)
        df_score = pd.concat([df_score, combined_df], ignore_index=True)

    prediction_path = PREDICTION_DIR / prediction_filename
    df_score.to_csv(prediction_path, index=False)
    print(f"Predictions saved to {prediction_path}")

    # Run Backtest Analysis
    df_backtest = pd.DataFrame(full_backtest_data)
    run_backtest_analysis(df_backtest)

    # --- Walk-Forward (Full History Retraining) ---
    if val_end_idx < len(data_files):
        print("\n--- Starting Walk-Forward Step (Single Step) ---")
        walk_end_idx = val_end_idx
        walk_pre_data_val = Path(data_files[walk_end_idx - 1]).stem
        
        # Create Walk Args
        walk_args = Args(data_start=train_start_idx, data_middle=walk_end_idx, data_end=walk_end_idx, pre_data=walk_pre_data_val)
        walk_args.save_name += "_walk"
        
        walk_model = eval(walk_args.model_name)(hidden_dim=walk_args.hidden_dim, num_heads=walk_args.num_heads, num_layers=walk_args.num_layers,
                                            out_features=walk_args.out_features, predictor_out_dim=args.target_dim,
                                            predictor_activation=walk_args.predictor_activation).to(walk_args.device)
        walk_model.load_state_dict(checkpoint['model'])

        # Create Walk Dataset (Train on everything seen so far)
        walk_dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), data_start=train_start_idx, data_middle=walk_end_idx, data_end=walk_end_idx)
        walk_loader = DataLoader(walk_dataset, batch_size=walk_args.batch_size, pin_memory=True, collate_fn=lambda x: x)

        walk_optimizer = optim.Adam(walk_model.parameters(), lr=walk_args.lr)
        walk_scheduler = StepLR(optimizer=walk_optimizer, step_size=5000, gamma=0.9, last_epoch=-1)

        print(f"Retraining on full history up to {walk_pre_data_val}...")
        for epoch in range(walk_args.max_epochs):
            loss = train_epoch(epoch, walk_args, walk_model, walk_loader, walk_optimizer, walk_scheduler, walk_args.loss_fcn)
            if (epoch+1) % 10 == 0:
                print(f"Walk Epoch {epoch+1}/{walk_args.max_epochs}, Loss: {loss:.6f}")

        # Save Walk Model
        walk_save_path = Path(walk_args.save_path) / f"{walk_pre_data_val}_walk_model.dat"
        torch.save({'model': walk_model.state_dict()}, walk_save_path)
        print(f"Walk-forward model saved to {walk_save_path}")

        # Predict Next Step
        if walk_end_idx < len(data_files):
            print("Predicting next step...")
            walk_inference_path = TRAIN_DATA_DIR / data_files[walk_end_idx]
            with open(walk_inference_path, "rb") as f:
                walk_sample = pickle.load(f)
            
            p_adj, n_adj, feats, _, _ = extract_data(walk_sample, walk_args.device)
            walk_model.eval()
            w_logits = walk_model(feats, p_adj, n_adj)
            w_preds = w_logits.detach().cpu().numpy()
            
            # Save results
            walk_code_file = data_code[walk_end_idx]
            walk_df_res = pd.read_csv(DAILY_STOCK_DIR / walk_code_file, dtype=object).reset_index(drop=True)
            
            # Format predictions
            if w_preds.ndim == 1: w_preds = w_preds[:, None]
            cols = ["score"] if w_preds.shape[1] == 1 else [f"score_t+{i+1}" for i in range(w_preds.shape[1])]
            w_scores_df = pd.DataFrame(w_preds, columns=cols)
            w_final_df = pd.concat([walk_df_res, w_scores_df], axis=1)
            
            walk_out_path = PREDICTION_DIR / walk_forward_filename
            w_final_df.to_csv(walk_out_path, index=False)
            print(f"Walk-forward prediction saved to {walk_out_path}")
            
    return df_score

if __name__ == "__main__":
    # Example usage: covering a wider range
    fun_train_predict(
        data_start=None, # Auto (0)
        data_middle=None, # Auto (End - Span)
        data_end=None,    # Auto (Max)
        train_start_date="2015-01-01",
        # Set validation to last year for a good backtest
        val_start_date="2024-01-01", 
        val_end_date="2025-12-31"
    )
