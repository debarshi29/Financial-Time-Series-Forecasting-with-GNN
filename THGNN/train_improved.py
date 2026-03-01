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
import torch.nn as nn
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

warnings.filterwarnings("ignore")
t_float = torch.float64
torch.multiprocessing.set_sharing_strategy('file_system')

BASE_DIR = Path(".").resolve()
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PREDICTION_DIR = DATA_DIR / "prediction"
PLOT_DIR = DATA_DIR / "plots"

for directory in [MODEL_DIR, PREDICTION_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_VALIDATION_SPAN = 4

# --- New Loss Functions ---
class CorrelationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        denom = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8
        loss = 1 - (cov / denom)
        return loss

class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=100.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.corr = CorrelationLoss()
        self.alpha = alpha # Weight for Correlation
        self.beta = beta   # Weight for Direction/Sign
        
    def forward(self, pred, target):
        # pred and target expected to be (N, 1) or (N, T, 1)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 1. MSE (Scale dependent)
        # Operating on 2D tensors directly is fine for MSE, but flattened is safer for sum reduction consistency
        loss_mse = self.mse(pred_flat, target_flat)
        
        
        # 2. Correlation Loss (Scale invariant, Shape dependent)
        # Needs flattened vectors to compute correlation over the batch/set
        # loss_corr = self.corr(pred_flat, target_flat)
        loss_corr = 0.0
        
        # 3. Soft Sign Loss (Direction dependent)
        # Operate element-wise on the original shape, then mean
        # loss_sign = torch.mean(torch.relu(-pred * target))
        loss_sign = 0.0
        
        total_loss = loss_mse # + (self.alpha * loss_corr) + (self.beta * loss_sign)
        return total_loss

class Args:
    def __init__(self, data_start, data_middle, data_end, pre_data, gpu=0):
        # device
        self.gpu = str(gpu)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # settings
        adj_threshold = 0.1
        self.adj_str = str(int(100*adj_threshold))
        self.data_start = data_start
        self.data_middle = data_middle
        self.data_end = data_end
        self.pre_data = pre_data
        
        # model settings
        self.hidden_dim = 128     
        self.num_layers = 2       
        self.num_heads = 4        
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.target_dim = 1
        self.predictor_activation = None
        
        # Training Settings
        self.max_epochs = 60
        self.lr = 0.0001
        self.save_path = str(MODEL_DIR)
        self.save_name = "Improved_" + self.model_name
        
        # Loss
        self.loss_fcn = HybridLoss(alpha=0.5, beta=0.5)

def train_improved(
    val_start_date="2024-01-01",
    val_end_date="2025-12-31"
):
    print("--- Starting Improved Training ---")
    
    # [Data Indexing Logic - shortened]
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    file_dates = [pd.to_datetime(Path(name).stem).normalize() for name in data_files]
    date_to_index = {dt: idx for idx, dt in enumerate(file_dates)}
    
    def get_idx(d_str):
        dt = pd.to_datetime(d_str).normalize()
        for idx, file_dt in enumerate(file_dates):
            if file_dt >= dt: return idx
        return len(data_files)-1

    val_start_idx = get_idx(val_start_date)
    val_end_idx = get_idx(val_end_date) + 1
    if val_end_idx > len(data_files): val_end_idx = len(data_files)
    
    # We train on everything BEFORE validation
    train_start_idx = 0
    # Assuming we want a decent training set
    
    # Pre data
    pre_data_value = Path(data_files[val_start_idx - 1]).stem
    
    args = Args(train_start_idx, val_start_idx, val_end_idx, pre_data_value)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    print(f"Training Range: 0 to {val_start_idx}")
    print(f"Validation Range: {val_start_idx} to {val_end_idx}")
    
    # Datasets
    dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), data_start=train_start_idx, data_middle=val_start_idx, data_end=val_end_idx)
    val_dataset = AllGraphDataSampler(str(TRAIN_DATA_DIR), mode="val", data_start=train_start_idx, data_middle=val_start_idx, data_end=val_end_idx)
    
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)
    
    # Infer dim
    _, _, _, sample_labels, _ = extract_data(dataset[0], args.device)
    args.target_dim = sample_labels.shape[-1] if sample_labels.dim() > 1 else 1

    # Model
    model = eval(args.model_name)(hidden_dim=args.hidden_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                                  out_features=args.out_features, predictor_out_dim=args.target_dim,
                                  predictor_activation=args.predictor_activation).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9)
    
    history = {'train_loss': [], 'val_loss': []}
    
    print("Training...")
    for epoch in range(args.max_epochs):
        model.train()
        total_loss = 0
        
        # Training Loop
        for batch_data in loader:
            for data in batch_data: # collate_fn=lambda x:x returns list
                optimizer.zero_grad()
                pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
                
                logits = model(features, pos_adj, neg_adj)
                
                # Apply hybrid loss
                # loss = args.loss_fcn(logits[mask], labels[mask])
                loss = torch.nn.L1Loss()(logits[mask], labels[mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        history['train_loss'].append(avg_loss)
        scheduler.step()
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_dataset: # Loop directly for simplicity
                   pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
                   logits = model(features, pos_adj, neg_adj)
                   # For validation we can check MSE separately or Hybrid
                   # Let's check pure MSE for "Accuracy" metric vs Hybrid
                   v_loss = args.loss_fcn(logits[mask], labels[mask]).item()
                   val_loss += v_loss
            
            avg_val = val_loss / len(val_dataset)
            history['val_loss'].append(avg_val)
            print(f"Epoch {epoch+1}: Train Loss {avg_loss:.6f} | Val Loss {avg_val:.6f}")
        else:
             print(f"Epoch {epoch+1}: Train Loss {avg_loss:.6f}")
             
    # Save Model
    save_path = Path(args.save_path) / f"IMPROVED_{pre_data_value}_epoch_{args.max_epochs}.dat"
    torch.save({'model': model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")
    
    # Plot Loss used
    plt.figure()
    plt.plot(history['train_loss'], label='Hybrid Train Loss')
    if history['val_loss']:
        # Align val loss points
        x_val = list(range(4, args.max_epochs, 5))
        plt.plot(x_val, history['val_loss'], label='Hybrid Val Loss')
    plt.legend()
    plt.savefig(PLOT_DIR / "improved_loss_curve.png")
    
if __name__ == "__main__":
    train_improved()
