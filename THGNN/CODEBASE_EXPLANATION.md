# THGNN Codebase Explanation & Architecture

This document provides a comprehensive technical overview of the **Temporal and Heterogeneous Graph Neural Network (THGNN)** repository. It covers the data processing pipeline, model architecture, and training workflow.

## 1. System Architecture Pipeline

The system operates in a sequential pipeline: gathering data, determining stock relationships, building graph snapshots, and finally training the model.

```mermaid
graph TD
    subgraph Data_Preparation ["Stage 1: Data Preparation"]
        direction TB
        Raw[("Yahoo Finance")] -->|download_market_data.py| MarketPKL["data/nifty50.pkl\n(Cleaned OHLCV + Technical Indicators)"]
        MarketPKL -->|generate_relation.py| Relations["data/relation/*.csv\n(Correlation Matrices)"]
    end

    subgraph Data_Generation ["Stage 2: Graph Generation"]
        direction TB
        MarketPKL & Relations -->|generate_data.py| GraphSnapshots["data/data_train_predict/<Date>.pkl\n(Features + Adjacency Tensors)"]
        MarketPKL -->|generate_data.py| DailyMeta["data/daily_stock/<Date>.csv\n(Metadata: Codes, Dates)"]
    end

    subgraph Modeling ["Stage 3: Training & Inference"]
        GraphSnapshots -->|data_loader.py| DataLoader
        DataLoader -->|train_ic_ranked.py| Model[("THGNN Model")]
        Model --> Trainer["trainer.py"]
        Trainer --> SavedModel["data/model_saved/*.dat"]
        Trainer --> Plots["data/plots/*.png"]
    end

    Data_Preparation --> Data_Generation
    Data_Generation --> Modeling
```

---

## 2. Directory Structure

| Directory | Key Files | Description |
| :--- | :--- | :--- |
| **`model/`** | `Thgnn.py`, `Thgnn_flexible.py` | Contains the PyTorch neural network definitions. `Thgnn_flexible.py` is the robust version that handles variable feature sizes. |
| **`utils/`** | `download_market_data.py` | Downloads stock data and calculates indicators (RSI, MACD, etc). |
| | `generate_relation.py` | Computes correlation between stocks to build the graph edges. |
| | `generate_data.py` | Fuses features and relations into daily graph snapshots for the model. |
| **`trainer/`** | `trainer.py` | Contains the training and evaluation loops (`train_epoch`, `eval_epoch`) and loss functions. |
| **`root`** | `train_ic_ranked.py` | **Primary training script.** Composite MSE + soft-Spearman IC + dispersion loss, early stopping, checkpoint saving. |
| | `data_loader.py` | Efficient `PyTorch Dataset` class for loading the pickled graph files. |

---

## 3. Model Architecture (THGNN)

The model is designed to capture both time-series trends (using GRU) and inter-stock relationships (using Heterogeneous GAT).

### Data Flow within the Model

1.  **Input**: Shape `(Batch, Sequence_Length, Features)`
2.  **Temporal Encoding**: Processed by a **GRU** to get a single vector representation per stock.
3.  **Spatial Aggregation**:
    *   **Positive GAT**: Aggregates info from positively correlated stocks.
    *   **Negative GAT**: Aggregates info from negatively correlated stocks.
4.  **Semantic Attention**: Learns how to weigh the importance of "Self" vs "Positive Neighbors" vs "Negative Neighbors".
5.  **Prediction**: An MLP produces the final score (e.g., predicted return).

```mermaid
graph LR
    subgraph Inputs
        Features["Stock Features (T time steps)"]
        PosAdj["Positive Adjacency Matrix"]
        NegAdj["Negative Adjacency Matrix"]
    end

    subgraph Temporal_Layer
        Features --> GRU[("GRU Layer")]
        GRU --> Hidden["Hidden State (H)"]
    end

    subgraph Spatial_Layers
        Hidden & PosAdj --> PosGAT["Positive GAT Head"]
        Hidden & NegAdj --> NegGAT["Negative GAT Head"]
        Hidden --> SelfFC["Self Linear Layer"]
    end

    subgraph Semantic_Aggregation
        PosGAT & NegGAT & SelfFC --> Stack["Stack Representations"]
        Stack --> SemAttn["Semantic Attention"]
        SemAttn --> FinalEmbed["Unified Embedding"]
    end

    FinalEmbed --> PairNorm["PairNorm"]
    PairNorm --> Predictor["MLP Predictor"]
    Predictor --> Output["Score / Class"]
```

---

## 4. Key Script Details

### `train_ic_ranked.py` (Primary Training Script)
This is the recommended entry point for all training.

*   **Date-Based Splits**: Specify `--train-start-date`, `--train-end-date`, etc. — indices are resolved automatically. No hard ceiling on training dates.
*   **Composite Loss**: Three terms, all O(1) in scale:
    1. `MSE / return_scale²` — normalized prediction error (`--return-scale 0.01` for decimal returns).
    2. `1 − soft Spearman IC` — cross-sectional ranking loss via differentiable soft ranks. Temperature anneals from 0.19 → 0.02 over training for stable gradients.
    3. Dimensionless spread-ratio penalty — penalizes both under-spreading (`pred_std < r_min × target_std`) and over-spreading (`pred_std > r_max × target_std`).
*   **IC Warmup**: IC weight ramps from 0 to `--ic-weight` over the first `--ic-warmup-epochs` (default 10) so MSE fits before ranking pressure is added. All three splits use the same ramped weight each epoch.
*   **Automatic Visualization**: Saves a loss curve to `data/plots/<pre_data>_icrank_loss_curve.png` after training. All three curves use the same loss objective per epoch, so overfitting/underfitting is directly visible.
*   **Early Stopping**: Stops after `--patience` epochs without validation rank IC improvement (default 20). Checkpoint is selected on exact Spearman IC, not composite loss.

### `model/Thgnn_flexible.py` vs `Thgnn.py`
*   **Standard (`Thgnn.py`)**: Has hardcoded input dimensions (often `in_features=6`). If your data has 7 columns (e.g., you added Moving Averages), it crashes.
*   **Flexible (`Thgnn_flexible.py`)**: During the first forward pass, it checks the input shape. If it doesn't match the GRU's expected size, it **automatically re-initializes** the GRU layer to fit the data. This makes it "plug-and-play" for different experiments.

---

## 5. How to Run (Workflow)

### Step 1: Get Data
```bash
# Download last 4 years of Nifty 50 data
python utils/download_market_data.py --start 2020-01-01 --end 2024-01-01
```

### Step 2: Build Relations
```bash
# Calculate correlations (who moves with whom?)
python utils/generate_relation.py --window 20
```

### Step 3: Create Graph Snapshots
```bash
# Create the .pkl files for PyTorch
python utils/generate_data.py --window 20 --horizon 1
```

### Step 4: Train
```bash
# Run training and see results
python train_ic_ranked.py
```
