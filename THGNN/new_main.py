from trainer.trainer import *
from data_loader import *
from model.Thgnn import *
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
DEBUG_EXTRACT_DATA = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_DATA_DIR = DATA_DIR / "data_train_predict"
DAILY_STOCK_DIR = DATA_DIR / "daily_stock"
MODEL_DIR = DATA_DIR / "model_saved"
PREDICTION_DIR = DATA_DIR / "prediction"
BACKTEST_DIR = DATA_DIR / "backtest_results"

for directory in [MODEL_DIR, PREDICTION_DIR, BACKTEST_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def extract_data_fixed(data, device):
    """
    Fixed version of extract_data that ensures features are properly shaped for GRU.
    Includes extensive debugging to identify data issues.
    """
    # Debug: Print raw data shapes
    if DEBUG_EXTRACT_DATA:
        print(f"\n=== DEBUG extract_data_fixed ===")
        print(f"Raw data['features'] shape: {data['features'].shape}")
        print(f"Raw data['features'] dtype: {data['features'].dtype}")
        print(f"Raw data['labels'] shape: {data['labels'].shape}")
        print(f"Raw data['pos_adj'] shape: {data['pos_adj'].shape}")
        print(f"Raw data['neg_adj'] shape: {data['neg_adj'].shape}")
    
    # Get tensors without aggressive squeezing
    pos_adj = data['pos_adj'].to(device)
    neg_adj = data['neg_adj'].to(device)
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    mask = data['mask']
    
    if DEBUG_EXTRACT_DATA:
        print(f"After .to(device):")
        print(f"  features shape: {features.shape}")
        print(f"  pos_adj shape: {pos_adj.shape}")
        print(f"  neg_adj shape: {neg_adj.shape}")
        print(f"  labels shape: {labels.shape}")
    
    # Check if features are empty in the source data
    if features.numel() == 0:
        raise ValueError(
            f"Features tensor is empty in the source data!\n"
            f"Shape: {features.shape}\n"
            f"This suggests the .pkl data files have no feature data.\n"
            f"Please check how the data files in {TRAIN_DATA_DIR} were created.\n"
            f"The 'features' field should contain the actual stock features (prices, volumes, etc.)"
        )
    
    # Handle adjacency matrices - these should be 2D (num_nodes, num_nodes)
    while pos_adj.dim() > 2:
        pos_adj = pos_adj.squeeze(0)
    while neg_adj.dim() > 2:
        neg_adj = neg_adj.squeeze(0)
    
    # Handle labels - squeeze extra batch dimensions but keep necessary dims
    while labels.dim() > 2:
        labels = labels.squeeze(0)
    
    # Handle features carefully - remove only outer batch wrappers.
    # Expected by model: (num_nodes, window, num_features)
    if features.dim() == 4 and features.size(0) == 1:
        features = features.squeeze(0)  # (num_nodes, window, num_features)
    elif features.dim() == 3 and features.size(0) == 1:
        features = features.squeeze(0)  # (num_nodes, num_features)
    
    # Now features should be 2D: (num_nodes, num_features)
    if features.dim() == 2:
        num_nodes, num_features = features.shape
        if DEBUG_EXTRACT_DATA:
            print(f"Features 2D: {num_nodes} nodes, {num_features} features")
        # Add sequence dimension for GRU: (num_nodes, 1, num_features)
        features = features.unsqueeze(1)
    elif features.dim() == 1:
        # Edge case: 1D tensor - could be single node or squeezed incorrectly
        num_features = features.size(0)
        if DEBUG_EXTRACT_DATA:
            print(f"Features 1D: {num_features} elements - treating as single node")
        features = features.unsqueeze(0).unsqueeze(0)  # (1, 1, num_features)
    elif features.dim() == 3:
        # Already 3D, expected format for GRU input.
        if DEBUG_EXTRACT_DATA:
            print(f"Features already 3D: {features.shape}")
        pass
    else:
        raise ValueError(f"Unexpected features dimension: {features.dim()}, shape: {features.shape}")
    
    if DEBUG_EXTRACT_DATA:
        print(f"Final features shape for GRU: {features.shape}")
        print(f"=== END DEBUG ===\n")
    
    # Final validation
    if features.size(-1) == 0:
        raise ValueError(
            f"Features have 0 elements in last dimension after processing!\n"
            f"Final shape: {features.shape}\n"
            f"This indicates the source data files are malformed."
        )
    
    return pos_adj, neg_adj, features, labels, mask

# Monkey patch the trainer module to use our fixed extract_data
import trainer.trainer as trainer_module
original_extract_data = trainer_module.extract_data
trainer_module.extract_data = extract_data_fixed


class Args:
    def __init__(self, gpu=0, subtask="regression"):
        self.gpu = str(gpu)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.max_epochs = 60
        self.epochs_eval = 10
        self.lr = 0.0002
        self.gamma = 0.3
        self.hidden_dim = 128
        self.num_heads = 8
        self.out_features = 32
        self.model_name = "StockHeteGAT"
        self.batch_size = 1
        self.loss_fcn = mse_loss
        self.target_dim = 1
        self.predictor_activation = None
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


class BacktestEngine:
    """Backtesting engine for stock predictions with portfolio management"""
    
    def __init__(self, initial_capital=100000, transaction_cost=0.001, top_n_stocks=10):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.top_n_stocks = top_n_stocks
        self.reset()
    
    def reset(self):
        self.cash = self.initial_capital
        self.holdings = {}
        self.portfolio_values = []
        self.dates = []
        self.trades = []
        self.returns = []
        
    def execute_strategy(self, predictions_df, date_col='date', ticker_col='code', 
                        score_col='score', price_col='close', return_col='label_regression'):
        """
        Execute trading strategy based on predictions
        
        Args:
            predictions_df: DataFrame with predictions and actual returns
            date_col: name of date column
            ticker_col: name of ticker/stock code column
            score_col: name of prediction score column
            price_col: name of price column
            return_col: name of actual return column (for calculating realized returns)
        """
        if predictions_df.empty:
            print("Warning: Empty predictions dataframe")
            return
        
        # Group by date
        for date in predictions_df[date_col].unique():
            daily_data = predictions_df[predictions_df[date_col] == date].copy()
            
            # Skip if no data
            if daily_data.empty:
                continue
            
            # Sort by prediction score (higher score = better predicted performance)
            daily_data = daily_data.sort_values(score_col, ascending=False)
            
            # Select top N stocks
            selected_stocks = daily_data.head(self.top_n_stocks)
            
            # Equal weight allocation
            allocation_per_stock = self.cash / len(selected_stocks)
            
            # Sell all current holdings
            for ticker in list(self.holdings.keys()):
                position = self.holdings[ticker]
                sell_value = position['shares'] * position['current_price']
                self.cash += sell_value * (1 - self.transaction_cost)
                self.trades.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'price': position['current_price'],
                    'value': sell_value
                })
                del self.holdings[ticker]
            
            # Buy new positions
            for _, row in selected_stocks.iterrows():
                ticker = row[ticker_col]
                price = float(row[price_col]) if pd.notna(row[price_col]) else 1.0
                
                if price <= 0:
                    continue
                
                shares = (allocation_per_stock * (1 - self.transaction_cost)) / price
                cost = shares * price
                
                if cost <= self.cash:
                    self.holdings[ticker] = {
                        'shares': shares,
                        'entry_price': price,
                        'current_price': price,
                        'expected_return': row[score_col] if pd.notna(row[score_col]) else 0
                    }
                    self.cash -= cost
                    
                    self.trades.append({
                        'date': date,
                        'ticker': ticker,
                        'action': 'BUY',
                        'shares': shares,
                        'price': price,
                        'value': cost
                    })
            
            # Update portfolio value
            holdings_value = sum([h['shares'] * h['current_price'] for h in self.holdings.values()])
            total_value = self.cash + holdings_value
            self.portfolio_values.append(total_value)
            self.dates.append(date)
            
            # Calculate returns
            if len(self.portfolio_values) > 1:
                daily_return = (total_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
                self.returns.append(daily_return)
            else:
                self.returns.append(0)
    
    def get_metrics(self):
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.array(self.returns)
        portfolio_values = np.array(self.portfolio_values)
        
        # Total return
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return (assuming daily data, 252 trading days)
        n_periods = len(returns)
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        
        return {
            'Total Return (%)': total_return * 100,
            'Annualized Return (%)': annualized_return * 100,
            'Volatility (%)': volatility * 100,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown * 100,
            'Calmar Ratio': calmar_ratio,
            'Win Rate (%)': win_rate * 100,
            'Total Trades': len(self.trades),
            'Final Portfolio Value': portfolio_values[-1],
            'Initial Capital': self.initial_capital
        }


def plot_backtest_results(backtest_engine, save_dir):
    """Generate comprehensive backtest visualization plots"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Equity Curve
    ax1 = axes[0, 0]
    dates = backtest_engine.dates
    portfolio_values = backtest_engine.portfolio_values
    
    ax1.plot(range(len(portfolio_values)), portfolio_values, linewidth=2, color='#2E86AB')
    ax1.axhline(y=backtest_engine.initial_capital, color='red', linestyle='--', 
                label='Initial Capital', alpha=0.7)
    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Trading Days', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    returns = np.array(backtest_engine.returns)
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
    ax2.plot(range(len(drawdown)), drawdown, linewidth=2, color='darkred')
    ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Trading Days', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns Distribution
    ax3 = axes[1, 0]
    ax3.hist(returns * 100, bins=50, alpha=0.7, color='#06A77D', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Return')
    ax3.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Daily Return (%)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Returns
    ax4 = axes[1, 1]
    cumulative_returns = (cumulative - 1) * 100
    ax4.plot(range(len(cumulative_returns)), cumulative_returns, linewidth=2, color='#F18F01')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Trading Days', fontsize=12)
    ax4.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'backtest_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Monthly returns heatmap (if enough data)
    if len(returns) > 20:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create monthly returns (simplified - group every 20 days as a month)
        monthly_returns = []
        for i in range(0, len(returns), 20):
            month_ret = np.prod(1 + returns[i:i+20]) - 1
            monthly_returns.append(month_ret * 100)
        
        months = [f'Period {i+1}' for i in range(len(monthly_returns))]
        colors = ['green' if r > 0 else 'red' for r in monthly_returns]
        
        ax.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Periodic Returns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Return (%)', fontsize=12)
        ax.set_xticks(range(len(monthly_returns)))
        ax.set_xticklabels(months, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'periodic_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {save_dir}")


def save_metrics_report(metrics, save_path):
    """Save metrics to a formatted text report"""
    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BACKTEST PERFORMANCE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 60 + "\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key:.<40} {value:>15.4f}\n")
            else:
                f.write(f"{key:.<40} {value:>15}\n")
        f.write("=" * 60 + "\n")
    
    print(f"Metrics report saved to {save_path}")


def fun_train_predict_backtest(
    data_start,
    data_middle,
    data_end,
    pre_data=None,
    prediction_horizon=None,
    target_dim=None,
    prediction_filename="pred.csv",
    walk_forward_filename="walk_forward_pred.csv",
    run_backtest=True,
    initial_capital=100000,
    transaction_cost=0.001,
    top_n_stocks=10
):
    """
    Extended training and prediction with backtesting capabilities
    """
    
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_DIR}")

    if pre_data is None:
        if data_middle <= 0 or data_middle > len(data_files):
            raise ValueError("data_middle index is out of range")
        pre_data_value = Path(data_files[data_middle - 1]).stem
    else:
        pre_data_value = pre_data

    globals()['pre_data'] = pre_data_value
    args = Args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Training phase
    dataset = AllGraphDataSampler(base_dir=str(TRAIN_DATA_DIR), data_start=data_start,
                                  data_middle=data_middle, data_end=data_end)
    val_dataset = AllGraphDataSampler(base_dir=str(TRAIN_DATA_DIR), mode="val", data_start=data_start,
                                      data_middle=data_middle, data_end=data_end)
    
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty")
    
    dataset_loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    val_dataset_loader = DataLoader(val_dataset, batch_size=1, pin_memory=True)

    # Use fixed extract_data function
    _, _, _, sample_labels, _ = extract_data_fixed(dataset[0], args.device)
    if sample_labels.dim() <= 1:
        inferred_dim = 1
    else:
        inferred_dim = sample_labels.shape[-1]

    if prediction_horizon is not None:
        if target_dim is not None and int(target_dim) != int(prediction_horizon):
            raise ValueError("prediction_horizon and target_dim must match")
        target_dim = int(prediction_horizon)

    if target_dim is None:
        target_dim = inferred_dim
    else:
        target_dim = int(target_dim)
        if target_dim != inferred_dim:
            raise ValueError(f"Target dim mismatch: {target_dim} vs {inferred_dim}")

    if target_dim < 1:
        raise ValueError("target_dim must be positive")

    args.target_dim = target_dim
    args.save_name = args.save_name + f"_tdim_{args.target_dim}"
    
    model = eval(args.model_name)(
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads,
        out_features=args.out_features,
        predictor_out_dim=args.target_dim,
        predictor_activation=args.predictor_activation
    ).to(args.device)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cold_scheduler = StepLR(optimizer=optimizer, step_size=5000, gamma=0.9, last_epoch=-1)
    
    print('=' * 60)
    print('TRAINING PHASE')
    print('=' * 60)
    epoch_pbar = tqdm(range(args.max_epochs), desc="Train epochs", unit="epoch")
    for epoch in epoch_pbar:
        train_loss = train_epoch(epoch=epoch, args=args, model=model, dataset_train=dataset_loader,
                                 optimizer=optimizer, scheduler=cold_scheduler, loss_fcn=args.loss_fcn)
        if epoch % args.epochs_eval == 0:
            eval_loss, _ = eval_epoch(args=args, model=model, dataset_eval=val_dataset_loader, loss_fcn=args.loss_fcn)
            print(f'Epoch: {epoch + 1}/{args.max_epochs}, train loss: {train_loss:.6f}, val loss: {eval_loss:.6f}')
            epoch_pbar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{eval_loss:.6f}")
        else:
            print(f'Epoch: {epoch + 1}/{args.max_epochs}, train loss: {train_loss:.6f}')
            epoch_pbar.set_postfix(train_loss=f"{train_loss:.6f}")
        
        if (epoch + 1) % args.epochs_save_by == 0:
            print("Saving model checkpoint...")
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, Path(args.save_path) / f"{pre_data_value}_epoch_{epoch + 1}.dat")

    # Predict
    print('\n' + '=' * 60)
    print('PREDICTION PHASE')
    print('=' * 60)
    checkpoint_path = Path(args.load_path) / f"{pre_data_value}_epoch_{epoch + 1}.dat"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    
    data_code = sorted([p.name for p in DAILY_STOCK_DIR.glob("*.csv")])
    data_code_last = data_code[data_middle:data_end]
    df_score = pd.DataFrame()
    
    for i in tqdm(range(len(val_dataset)), desc="Generating predictions", unit="day"):
        df = pd.read_csv(DAILY_STOCK_DIR / data_code_last[i], dtype=object)
        tmp_data = val_dataset[i]
        # Use fixed extract_data function
        pos_adj, neg_adj, features, labels, mask = extract_data_fixed(tmp_data, args.device)
        model.eval()
        logits = model(features, pos_adj, neg_adj)
        predictions = logits.detach().cpu().numpy()
        
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        
        if predictions.shape[1] == 1:
            score_columns = ["score"]
        else:
            score_columns = [f"score_t+{idx + 1}" for idx in range(predictions.shape[1])]
        
        scores_df = pd.DataFrame(predictions, columns=score_columns)
        df = df.reset_index(drop=True)
        combined_df = pd.concat([df, scores_df], axis=1)
        df_score = pd.concat([df_score, combined_df], ignore_index=True)

    prediction_path = PREDICTION_DIR / prediction_filename
    df_score.to_csv(prediction_path, index=False)
    print(f"Saved predictions to {prediction_path.resolve()}")

    # Walk-forward prediction
    print('\n' + '=' * 60)
    print('WALK-FORWARD RETRAINING')
    print('=' * 60)
    
    if data_end <= data_start:
        raise ValueError("data_end must be greater than data_start")

    walk_pre_data_value = Path(data_files[data_end - 1]).stem
    original_globals = {
        "pre_data": globals().get("pre_data"),
        "data_middle": globals().get("data_middle"),
        "data_end": globals().get("data_end"),
    }
    
    globals()['pre_data'] = walk_pre_data_value
    globals()['data_middle'] = data_end
    globals()['data_end'] = data_end

    walk_args = Args()
    walk_args.target_dim = args.target_dim
    walk_args.save_name = walk_args.save_name + f"_tdim_{walk_args.target_dim}"
    
    walk_model = eval(walk_args.model_name)(
        hidden_dim=walk_args.hidden_dim,
        num_heads=walk_args.num_heads,
        out_features=walk_args.out_features,
        predictor_out_dim=walk_args.target_dim,
        predictor_activation=walk_args.predictor_activation
    ).to(walk_args.device)
    walk_model.load_state_dict(checkpoint['model'])

    walk_dataset = AllGraphDataSampler(base_dir=str(TRAIN_DATA_DIR), data_start=data_start,
                                       data_middle=data_end, data_end=data_end)
    if len(walk_dataset) == 0:
        raise RuntimeError("Walk-forward dataset is empty")
    
    walk_loader = DataLoader(walk_dataset, batch_size=walk_args.batch_size, pin_memory=True, collate_fn=lambda x: x)
    walk_optimizer = optim.Adam(walk_model.parameters(), lr=walk_args.lr)
    walk_scheduler = StepLR(optimizer=walk_optimizer, step_size=5000, gamma=0.9, last_epoch=-1)

    walk_epoch_pbar = tqdm(range(walk_args.max_epochs), desc="Walk-forward epochs", unit="epoch")
    for epoch in walk_epoch_pbar:
        train_loss = train_epoch(epoch=epoch, args=walk_args, model=walk_model, dataset_train=walk_loader,
                                 optimizer=walk_optimizer, scheduler=walk_scheduler, loss_fcn=walk_args.loss_fcn)
        print(f'Walk-forward Epoch: {epoch + 1}/{walk_args.max_epochs}, train loss: {train_loss:.6f}')
        walk_epoch_pbar.set_postfix(train_loss=f"{train_loss:.6f}")
        
        if (epoch + 1) % walk_args.epochs_save_by == 0:
            print("Saving walk-forward model...")
            state = {'model': walk_model.state_dict(), 'optimizer': walk_optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(state, Path(walk_args.save_path) / f"{walk_pre_data_value}_walk_epoch_{epoch + 1}.dat")

    walk_checkpoint_path = Path(walk_args.load_path) / f"{walk_pre_data_value}_walk_epoch_{epoch + 1}.dat"
    walk_checkpoint = torch.load(walk_checkpoint_path)
    walk_model.load_state_dict(walk_checkpoint['model'])

    globals()['pre_data'] = original_globals["pre_data"]
    globals()['data_middle'] = original_globals["data_middle"]
    globals()['data_end'] = original_globals["data_end"]

    walk_combined_df = None
    walk_inference_index = data_end
    if walk_inference_index >= len(data_files):
        print("Walk-forward inference skipped: no sample exists beyond data_end.")
    else:
        walk_inference_path = TRAIN_DATA_DIR / data_files[walk_inference_index]
        with open(walk_inference_path, "rb") as inference_file:
            walk_sample = pickle.load(inference_file)

        # Use fixed extract_data function
        pos_adj, neg_adj, features, labels, mask = extract_data_fixed(walk_sample, walk_args.device)
        walk_model.eval()
        walk_logits = walk_model(features, pos_adj, neg_adj)
        walk_predictions = walk_logits.detach().cpu().numpy()
        
        if walk_predictions.ndim == 1:
            walk_predictions = walk_predictions[:, None]
        
        if walk_predictions.shape[1] == 1:
            walk_score_columns = ["score"]
        else:
            walk_score_columns = [f"score_t+{idx + 1}" for idx in range(walk_predictions.shape[1])]

        walk_scores_df = pd.DataFrame(walk_predictions, columns=walk_score_columns)

        if walk_inference_index >= len(data_code):
            print("Walk-forward CSV output skipped: index out of range.")
        else:
            walk_data_code = data_code[walk_inference_index]
            walk_df = pd.read_csv(DAILY_STOCK_DIR / walk_data_code, dtype=object).reset_index(drop=True)
            walk_combined_df = pd.concat([walk_df, walk_scores_df], axis=1)
            walk_prediction_path = PREDICTION_DIR / walk_forward_filename
            walk_combined_df.to_csv(walk_prediction_path, index=False)
            print(f"Saved walk-forward predictions to {walk_prediction_path.resolve()}")

    # Backtesting
    if run_backtest:
        print('\n' + '=' * 60)
        print('BACKTESTING PHASE')
        print('=' * 60)
        
        # Initialize backtest engine
        backtest = BacktestEngine(
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            top_n_stocks=top_n_stocks
        )
        
        # Prepare data for backtesting
        print(f"Predictions DataFrame shape: {df_score.shape}")
        print(f"Available columns: {df_score.columns.tolist()}")
        
        # Map common column name variations
        column_mapping = {
            'date': ['date', 'Date', 'DATE', 'dt', 'DT', 'trading_date', 'trade_date'],
            'code': ['code', 'Code', 'CODE', 'ticker', 'Ticker', 'symbol', 'Symbol', 'stock_code'],
            'score': ['score', 'Score', 'SCORE', 'prediction', 'pred', 'score_t+1'],
            'close': ['close', 'Close', 'CLOSE', 'close_price', 'price', 'adj_close']
        }
        
        actual_columns = {}
        for target_col, possible_names in column_mapping.items():
            found = False
            for col_name in possible_names:
                if col_name in df_score.columns:
                    actual_columns[target_col] = col_name
                    found = True
                    break
            if not found:
                actual_columns[target_col] = None
        
        print(f"Mapped columns: {actual_columns}")
        
        # If close price is missing, try to merge from daily stock data
        if actual_columns['close'] is None and actual_columns['date'] is not None and actual_columns['code'] is not None:
            print("\nClose price not found in predictions. Attempting to merge from daily stock files...")
            
            # Read all daily stock files and extract close prices
            all_stock_data = []
            for csv_file in sorted(DAILY_STOCK_DIR.glob("*.csv")):
                try:
                    daily_df = pd.read_csv(csv_file)
                    # Try to find close price column
                    close_col = None
                    for col in ['close', 'Close', 'CLOSE', 'adj_close', 'Adj_Close']:
                        if col in daily_df.columns:
                            close_col = col
                            break
                    
                    if close_col:
                        # Extract relevant columns
                        date_col = None
                        code_col = None
                        for col in ['dt', 'date', 'Date', 'DATE']:
                            if col in daily_df.columns:
                                date_col = col
                                break
                        for col in ['code', 'Code', 'CODE', 'ticker']:
                            if col in daily_df.columns:
                                code_col = col
                                break
                        
                        if date_col and code_col:
                            subset = daily_df[[code_col, date_col, close_col]].copy()
                            subset.columns = ['code', 'date', 'close']
                            all_stock_data.append(subset)
                except Exception as e:
                    print(f"Warning: Could not read {csv_file.name}: {e}")
                    continue
            
            if all_stock_data:
                stock_prices = pd.concat(all_stock_data, ignore_index=True)
                # Convert types
                stock_prices['close'] = pd.to_numeric(stock_prices['close'], errors='coerce')
                
                # Merge with predictions
                df_score_backup = df_score.copy()
                df_score = df_score.merge(
                    stock_prices,
                    left_on=[actual_columns['code'], actual_columns['date']],
                    right_on=['code', 'date'],
                    how='left'
                )
                
                if 'close' in df_score.columns:
                    actual_columns['close'] = 'close'
                    print(f"Successfully merged close prices. New shape: {df_score.shape}")
                else:
                    print("Merge failed. Using backup DataFrame.")
                    df_score = df_score_backup

        # Fallback: build synthetic close prices from the processed market dataset.
        # nifty50.pkl stores daily close returns; convert each ticker to a price index.
        if actual_columns['close'] is None and actual_columns['date'] is not None and actual_columns['code'] is not None:
            market_path = DATA_DIR / "nifty50.pkl"
            if market_path.exists():
                print("\nClose price still missing. Building synthetic close from data/nifty50.pkl ...")
                try:
                    market_df = pd.read_pickle(market_path)
                    market_df = pd.DataFrame(market_df).copy()
                    market_df["dt"] = pd.to_datetime(market_df["dt"]).dt.strftime("%Y-%m-%d")
                    market_df["close"] = pd.to_numeric(market_df["close"], errors="coerce")
                    market_df = market_df.dropna(subset=["code", "dt", "close"])
                    market_df = market_df.sort_values(["code", "dt"])
                    market_df["close"] = (
                        market_df.groupby("code")["close"].transform(lambda s: 100.0 * (1.0 + s).cumprod())
                    )
                    synthetic_close = market_df[["code", "dt", "close"]].drop_duplicates()

                    df_score_backup = df_score.copy()
                    df_score = df_score.merge(
                        synthetic_close,
                        left_on=[actual_columns["code"], actual_columns["date"]],
                        right_on=["code", "dt"],
                        how="left",
                    )
                    if "close" in df_score.columns:
                        actual_columns["close"] = "close"
                        print(f"Successfully merged synthetic close prices. New shape: {df_score.shape}")
                    else:
                        print("Synthetic merge did not produce close column. Using backup DataFrame.")
                        df_score = df_score_backup
                except Exception as e:
                    print(f"Warning: Failed to build synthetic close prices: {e}")
        
        # Check if we have minimum required columns
        missing_required = [k for k, v in actual_columns.items() if v is None]
        
        if missing_required:
            print(f"\nERROR: Cannot run backtest - missing required columns: {missing_required}")
            print("\nDataFrame preview:")
            print(df_score.head())
            print("\nSkipping backtesting. Please ensure your data has columns for:")
            print("  - date/Date/dt (trading date)")
            print("  - code/Code/ticker (stock identifier)")
            print("  - score/Score/score_t+1 (model prediction)")
            print("  - close/Close (closing price)")
            print("\nNote: You can add close prices to your daily stock CSV files,")
            print("      or the script will attempt to merge them automatically.")
            return df_score, walk_combined_df, None, None
        
        # Run backtest with actual column names
        print(f"\nRunning backtest with columns: {actual_columns}")
        backtest.execute_strategy(
            df_score,
            date_col=actual_columns['date'],
            ticker_col=actual_columns['code'],
            score_col=actual_columns['score'],
            price_col=actual_columns['close']
        )
        
        # Calculate metrics
        metrics = backtest.get_metrics()
        
        # Print metrics
        print('\n' + '=' * 60)
        print('BACKTEST RESULTS')
        print('=' * 60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:>15.4f}")
            else:
                print(f"{key:.<40} {value:>15}")
        print('=' * 60)
        
        # Generate plots
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backtest_results_dir = BACKTEST_DIR / f"backtest_{timestamp}"
        plot_backtest_results(backtest, backtest_results_dir)
        
        # Save metrics report
        metrics_path = backtest_results_dir / 'metrics_report.txt'
        save_metrics_report(metrics, metrics_path)
        
        # Save trades log
        trades_df = pd.DataFrame(backtest.trades)
        if not trades_df.empty:
            trades_path = backtest_results_dir / 'trades_log.csv'
            trades_df.to_csv(trades_path, index=False)
            print(f"Trades log saved to {trades_path}")
        
        # Save portfolio values
        portfolio_df = pd.DataFrame({
            'date': backtest.dates,
            'portfolio_value': backtest.portfolio_values,
            'return': [0] + backtest.returns
        })
        portfolio_path = backtest_results_dir / 'portfolio_values.csv'
        portfolio_df.to_csv(portfolio_path, index=False)
        print(f"Portfolio values saved to {portfolio_path}")
        
        print(f"\nAll backtest results saved to {backtest_results_dir}")
        
        return df_score, walk_combined_df, backtest, metrics
    
    return df_score, walk_combined_df, None, None


if __name__ == "__main__":
    # Train up to 2023-12-29 and validate from the next trading day onward.
    data_files = sorted([p.name for p in TRAIN_DATA_DIR.glob("*.pkl")])
    if not data_files:
        raise RuntimeError(f"No training data found in {TRAIN_DATA_DIR}")

    file_dates = [Path(name).stem for name in data_files]
    target_train_end = "2023-12-29"
    target_val_start = "2024-01-01"

    if target_val_start in file_dates:
        data_middle = file_dates.index(target_val_start)
    else:
        # Fallback to first date after train end.
        data_middle = file_dates.index(target_train_end) + 1
    data_start = 0
    data_end = len(data_files)

    # Run training, prediction, and backtesting.
    predictions, walk_forward_preds, backtest_engine, metrics = fun_train_predict_backtest(
        data_start=data_start,
        data_middle=data_middle,
        data_end=data_end,
        pre_data=target_train_end,
        run_backtest=True,
        initial_capital=100000,
        transaction_cost=0.001,
        top_n_stocks=10
    )
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
