import torch
import torch.nn as nn


def _squeeze_trailing_dim(tensor):
    """Remove trailing dimensions of size 1 without touching leading axes."""
    while tensor.dim() > 0 and tensor.size(-1) == 1:
        tensor = tensor.squeeze(-1)
    return tensor


def mse_loss(logits, targets):
    mse = nn.MSELoss()
    logits = _squeeze_trailing_dim(logits)
    targets = _squeeze_trailing_dim(targets)
    loss = mse(logits, targets)
    return loss


def bce_loss(logits, targets):
    bce = nn.BCELoss()
    logits = _squeeze_trailing_dim(logits)
    targets = _squeeze_trailing_dim(targets)
    loss = bce(logits, targets)
    return loss


def evaluate(model, features, adj_pos, adj_neg, labels, mask, loss_func=nn.L1Loss()):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj_pos, adj_neg)

    loss = loss_func(logits,labels)
    return loss, logits


def extract_data(data_dict, device):
    pos_adj = data_dict['pos_adj'].to(device)
    neg_adj = data_dict['neg_adj'].to(device)
    features = data_dict['features'].to(device)
    labels = data_dict['labels'].to(device)
    
    # Only squeeze the batch dimension if it exists and is 1?
    # Actually, the original code used squeeze() which destroys all dim-1s.
    # We really only want to remove the 'extra' batch wrapper if it exists?
    # But wait, the data comes from pickle as (NumStocks, Win, Feat).
    # If the DataLoader adds a batch dim -> (1, NumStocks, Win, Feat).
    # We probably want to revert to (NumStocks, Win, Feat).
    
    # Safe approach: Squeeze ONLY the first dimension if it's 1, up to a point.
    # But features might be (1, 20, 6) for 1 stock.
    # The Model expects (Batch=NumStocks, Seq=20, Feat=6).
    # So we SHOULD NOT squeeze features if it results in < 3 dims.
    
    if features.dim() > 3 and features.size(0) == 1:
        features = features.squeeze(0)
    
    # If features was (1, 20, 6), squeeze() made it (20, 6).
    # We want to KEEP it (1, 20, 6).
    # So we actually should NOT squeeze features generally.
    
    # Original 'pos_adj' and 'neg_adj' logic might benefit from squeeze if they are adjacency matrices.
    # But let's check their usage. They are used in graph attention.
    
    # Helper to safe-squeeze
    def safe_squeeze(t):
        if t.dim() > 0 and t.size(0) == 1:
            return t.squeeze(0)
        return t

    # Adjacency matrices usually (Batch, N, N). If Batch=1 -> (N, N).
    if pos_adj.dim() > 2: pos_adj = safe_squeeze(pos_adj)
    if neg_adj.dim() > 2: neg_adj = safe_squeeze(neg_adj)
    
    # Labels usually (NumStocks, 1).
    if labels.dim() > 1 and labels.size(-1) == 1:
         labels = labels.squeeze(-1) # Make (NumStocks,)
    
    mask = data_dict['mask']
    return pos_adj, neg_adj, features, labels, mask


def train_epoch(epoch, args, model, dataset_train, optimizer, scheduler, loss_fcn):
    model.train()
    loss_return = 0
    valid_steps = 0
    for batch_data in dataset_train:
        for batch_idx, data in enumerate(batch_data):
            model.zero_grad()
            try:
                pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
            except ValueError as exc:
                print(f"SKIP BAD TRAIN SAMPLE: {exc}")
                continue
            
            # DEBUG: Catch bad shapes
            if features.dim() != 3:
                print(f"SKIP ERROR: Found bad feature shape: {features.shape}. Expected 3D (Batch, Seq, Feat).")
                continue # Skip this batch
                
            logits = model(features, pos_adj, neg_adj)
            loss = loss_fcn(logits[mask], labels[mask])
            loss.backward()
            optimizer.step()
            scheduler.step()
            if batch_idx == 0:
                loss_return += loss.data
                valid_steps += 1
    if valid_steps == 0:
        return 0.0
    return loss_return / valid_steps


def eval_epoch(args, model, dataset_eval, loss_fcn):
    total_loss = 0.0
    logits = None
    num_batches = 0
    for data in dataset_eval:
        try:
            pos_adj, neg_adj, features, labels, mask = extract_data(data, args.device)
        except ValueError as exc:
            print(f"SKIP BAD EVAL SAMPLE: {exc}")
            continue
        batch_loss, logits = evaluate(model, features, pos_adj, neg_adj, labels, mask, loss_func=loss_fcn)
        total_loss += batch_loss.item() if hasattr(batch_loss, "item") else float(batch_loss)
        num_batches += 1
    if num_batches == 0:
        return 0.0, logits
    return total_loss / num_batches, logits
