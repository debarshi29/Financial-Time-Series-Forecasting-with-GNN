"""
Hybrid THGNN × MaGNet model — Cascaded-Parallel design.

Architecture:
  Input (N, T, F)
    → LayerNorm + cross-sectional demean
    → Linear(F → D)                                [feature embedding]
    → MAGEBlock × num_mage_layers                  [MaGNet temporal encoder]
        BiGRU(fwd+bwd) → Gating → SparseMoE → MHA → Z_temp (N, T, D)
    │
    ├─── Path A: TCH(Z_temp)            → h_causal (N, D)  [lead-lag causality]
    │    Flatten (T·N, D) → Causal MHA (block mask) → ReTanh FFN
    │    → H_TCH incidence matrix → Hypergraph conv → slice last timestep
    │
    └─── Path B: h_temp = Z_temp[:, -1, :]         (N, D)
            ├─── PosGAT(h_temp, pos_adj)  → h_pos  (N, D)  [explicit co-movement]
            ├─── NegGAT(h_temp, neg_adj)  → h_neg  (N, D)  [explicit inverse-movement]
            └─── GPHypergraph(h_temp)     → h_gph  (N, D)  [latent macro themes]

  4-stream Semantic Attention Fusion(h_causal, h_pos, h_neg, h_gph)
    → PairNorm-SI
    → Linear(D → 1)                                (N, 1) predictions

Design rationale:
- TCH reinstated: captures asynchronous lead-lag causal ripples that the MAGE temporal
  MHA misses because MHA fuses all timesteps per stock, not across stocks and time.
- TCH and GPH/GAT are kept strictly parallel (Path A vs Path B) so each discovers an
  orthogonal market force on pristine, un-mixed representations.
- 2D Feature Attention dropped: MAGE's MHA already captures cross-timestep deps per stock.
- IC-ranked composite loss (MSE + Spearman IC + dispersion) from THGNN kept — better
  for cross-sectional portfolio ranking than MaGNet's original BCE.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# ---------------------------------------------------------------------------
# Reused from THGNN (self-contained copies)
# ---------------------------------------------------------------------------

class GraphAttnMultiHead(Module):
    """Multi-head additive graph attention (THGNN)."""

    def __init__(self, in_features, out_features, negative_slope=0.2,
                 num_heads=4, bias=True, residual=True):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        self.project = nn.Linear(in_features, num_heads * out_features) if residual else None
        self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features)) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        stdv2 = 1.0 / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv2, stdv2)
        self.weight_v.data.uniform_(-stdv2, stdv2)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(1, 0, 2)
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        weight = self.leaky_relu(f_1 + f_2)           # (heads, N, N)
        adj_exp = adj_mat.unsqueeze(0)                 # (1, N, N)
        # Dense masked softmax: scale logits by adj weight (same as original mul),
        # then mask zero entries to -inf so they don't participate in softmax.
        # Preserves exact semantics of the original sparse softmax for any adj values.
        logits = weight * adj_exp
        logits = logits.masked_fill(adj_exp == 0, float("-inf"))
        attn = torch.softmax(logits, dim=2)
        attn = torch.nan_to_num(attn, nan=0.0)         # isolated nodes → 0
        support = torch.matmul(attn, support)
        support = support.permute(1, 0, 2).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        return support, (attn if requires_weight else None)


class PairNorm(nn.Module):
    """PairNorm to prevent over-smoothing in GNNs (THGNN)."""

    def __init__(self, mode="PN-SI", scale=1):
        assert mode in ("None", "PN", "PN-SI", "PN-SCS")
        super().__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == "None":
            return x
        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            return self.scale * x / rownorm_mean
        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_ind = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            return self.scale * x / rownorm_ind
        # PN-SCS
        rownorm_ind = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
        return self.scale * x / rownorm_ind - col_mean


class GraphAttnSemIndividual(Module):
    """Semantic attention over multiple embedding streams (THGNN)."""

    def __init__(self, in_features, hidden_size=128, act=None):
        super().__init__()
        act = act if act is not None else nn.Tanh()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            act,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, inputs, requires_weight=False):
        # inputs: (N, num_streams, D)
        w = self.project(inputs)              # (N, num_streams, 1)
        beta = torch.softmax(w, dim=1)        # (N, num_streams, 1)
        out = (beta * inputs).sum(dim=1)      # (N, D)
        return out, (beta if requires_weight else None)


# ---------------------------------------------------------------------------
# Shared activation
# ---------------------------------------------------------------------------

def _retanh(x: torch.Tensor) -> torch.Tensor:
    """ReTanh: 0 for x≤0, tanh(x) for x>0.
    Combines ReLU sparsity with tanh boundedness — keeps hyperedge
    assignments sparse while bounding them for numerical stability.
    """
    return torch.where(x <= 0, torch.zeros_like(x), torch.tanh(x))


# ---------------------------------------------------------------------------
# MaGNet components
# ---------------------------------------------------------------------------

class SparseMoE(nn.Module):
    """Top-1 Sparse Mixture-of-Experts layer (MaGNet §3.3.3).

    Each token is routed to exactly one expert via argmax of the gating
    softmax. All experts are computed (ensuring full gradient flow) and
    the selected expert's output is weighted by its routing probability
    for load-balanced training signal.
    """

    def __init__(self, embed_dim: int, num_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        D, E = embed_dim, num_experts
        self.gate = nn.Linear(D, E)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(D, D),
            )
            for _ in range(E)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)
        N, T, D = x.shape
        E = len(self.experts)
        x_flat = x.reshape(N * T, D)                           # (K, D)

        probs = torch.softmax(self.gate(x_flat), dim=-1)       # (K, E)
        assignments = probs.argmax(dim=-1)                     # (K,)

        expert_outs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=-1
        )                                                       # (K, D, E)

        idx = assignments.view(N * T, 1, 1).expand(N * T, D, 1)
        selected = expert_outs.gather(dim=2, index=idx).squeeze(2)  # (K, D)

        w = probs.gather(dim=1, index=assignments.unsqueeze(1))     # (K, 1)
        output = w * selected                                        # (K, D)

        return output.reshape(N, T, D)


class MAGEBlock(nn.Module):
    """Mamba-Attention-Gating-Experts block (MaGNet §3.3), BiGRU-lite variant.

    BiGRU(fwd + bwd) → Gating → SparseMoE → Multi-head self-attention.
    Replaces the Mamba SSM with a bidirectional GRU, eliminating the
    mamba-ssm dependency while preserving the same information-flow design.
    """

    def __init__(self, embed_dim: int, num_experts: int = 4,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        D = embed_dim

        self.bigru = nn.GRU(
            input_size=D, hidden_size=D,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.gate_fwd = nn.Linear(D, D, bias=True)
        self.gate_bwd = nn.Linear(D, D, bias=False)
        self.norm_gru = nn.LayerNorm(D)

        self.moe = SparseMoE(embed_dim=D, num_experts=num_experts, dropout=dropout)
        self.norm_moe = nn.LayerNorm(D)

        self.mha = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_mha = nn.LayerNorm(D)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)

        gru_out, _ = self.bigru(x)                         # (N, T, 2D)
        D = x.size(-1)
        z_fwd = gru_out[:, :, :D]
        z_bwd = gru_out[:, :, D:]

        gate = torch.sigmoid(self.gate_fwd(z_fwd) + self.gate_bwd(z_bwd))
        z_G = gate * z_fwd + (1.0 - gate) * z_bwd
        z_G = self.norm_gru(self.drop(z_G) + x)

        z_moe = self.moe(z_G)
        z_moe = self.norm_moe(self.drop(z_moe) + z_G)

        z_attn, _ = self.mha(z_moe, z_moe, z_moe)
        z_out = self.norm_mha(self.drop(z_attn) + z_moe)

        return z_out                                        # (N, T, D)


class TemporalCausalHypergraph(nn.Module):
    """Temporal-Causal Hypergraph (TCH) from MaGNet §3.5.1.

    Discovers asynchronous lead-lag causal relationships by treating every
    (time, stock) pair as an independent node and attending causally across
    the full T×N spatiotemporal grid.

    Flow:
      Z_temp (N, T, D)
        → flatten to Z_flat (T·N, D)  [each row = one (time, stock) node]
        → Causal MHA with upper-triangular block mask
          [node at time t attends only to t' ≤ t — no future leakage]
        → two-layer ReTanh FFN → H_TCH (T·N, M1) soft incidence matrix
          [each column = one causal hyperedge over all (time, stock) nodes]
        → efficient hypergraph conv: Z' = ELU(H·(H^T·Proj(Z))) + Z  [O(T·N·M1·D)]
        → reshape (N, T, D) → slice final timestep → h_causal (N, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_hyper_edges: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        D, M1 = embed_dim, num_hyper_edges

        self.causal_mha = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_mha = nn.LayerNorm(D)

        # Two-layer FFN to build incidence matrix from causally-shaped features
        self.to_hyper = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, M1),
        )

        self.proj = nn.Linear(D, D, bias=False)
        self.norm_out = nn.LayerNorm(D)
        self.drop = nn.Dropout(dropout)

        self._mask_TN: tuple[int, int] | None = None
        self._mask_cache: torch.Tensor | None = None

    def forward(self, z_temp: torch.Tensor) -> torch.Tensor:
        # z_temp: (N, T, D)
        N, T, D = z_temp.shape
        TN = T * N

        # Flatten: (N, T, D) → ordered (t0·S0, …, t0·SN-1, t1·S0, …)
        z_flat = z_temp.permute(1, 0, 2).reshape(TN, D)     # (T·N, D)

        # Causal MHA: each (t, s) node can only attend to nodes at t' ≤ t
        causal_mask = self._build_causal_mask(T, N, z_flat.device)
        z_seq = z_flat.unsqueeze(0)                          # (1, T·N, D)
        z_attn, _ = self.causal_mha(
            z_seq, z_seq, z_seq,
            attn_mask=causal_mask,
            need_weights=False,
        )
        z_attn = z_attn.squeeze(0)                           # (T·N, D)
        z_attn = self.norm_mha(self.drop(z_attn) + z_flat)  # residual

        # Build sparse incidence matrix via ReTanh (bounds + sparsity without softmax)
        H_TCH = _retanh(self.to_hyper(z_attn))              # (T·N, M1)

        # Efficient hypergraph convolution — avoid materialising (T·N × T·N):
        #   Z' = ELU( H · (H^T · Proj(Z)) )  →  O(T·N · M1 · D)
        z_proj = self.proj(z_attn)                           # (T·N, D)
        HT_z = H_TCH.t() @ z_proj                           # (M1, D)
        z_out = F.elu(H_TCH @ HT_z)                         # (T·N, D)
        z_out = self.norm_out(self.drop(z_out) + z_attn)    # residual

        # Reshape back and return the causally-enriched final-timestep state
        z_out = z_out.reshape(T, N, D).permute(1, 0, 2)     # (N, T, D)
        return z_out[:, -1, :]                               # h_causal: (N, D)

    def _build_causal_mask(self, T: int, N: int, device: torch.device) -> torch.Tensor:
        if self._mask_TN == (T, N) and self._mask_cache is not None:
            return self._mask_cache
        TN = T * N
        idx = torch.arange(TN, device=device)
        t_idx = idx // N
        future = t_idx.unsqueeze(1) < t_idx.unsqueeze(0)    # (TN, TN) bool
        mask = torch.zeros(TN, TN, device=device)
        mask[future] = float("-inf")
        self._mask_TN = (T, N)
        self._mask_cache = mask
        return mask


class GPHypergraph(nn.Module):
    """Global Probabilistic Hypergraph (MaGNet §3.5.2).

    Learns soft hyperedge assignments and weights each hyperedge by its
    uniqueness (Jensen-Shannon Divergence vs other hyperedges).

    Convolution: Z' = ELU(H · diag(w) · H^T · Proj(h)) + h  (residual)
    where H ∈ (N, M) is the soft incidence matrix (column-softmax),
          W = diag(w_1, …, w_M) is the JSD-based importance weighting.
    """

    def __init__(self, embed_dim: int, num_hyper_edges: int = 32):
        super().__init__()
        D, M = embed_dim, num_hyper_edges
        self.to_hyper = nn.Linear(D, M)
        self.proj = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, D)

        H_raw = self.to_hyper(h)                    # (N, M)
        H_raw = _retanh(H_raw)
        H_GPH = torch.softmax(H_raw, dim=0)         # column-softmax → prob dist over stocks

        w = self._jsd_weights(H_GPH)                # (M,)

        # Efficient conv: H · diag(w) · (H^T · Proj(h))
        z_proj = self.proj(h)                        # (N, D)
        HT_z = H_GPH.t() @ z_proj                   # (M, D)
        HW_HT_z = H_GPH @ (w.unsqueeze(1) * HT_z)  # (N, D)
        z = F.elu(HW_HT_z)

        return self.norm(z + h)

    @staticmethod
    def _jsd_weights(H_GPH: torch.Tensor) -> torch.Tensor:
        """Per-hyperedge importance via average Jensen-Shannon Divergence.

        Hyperedges more distinct from the others receive higher importance.
        Z-score normalisation + softmax keeps weights bounded and differentiable.
        """
        eps = 1e-8
        H_t = H_GPH.t()                                    # (M, N)
        M = H_t.size(0)

        p = H_t.unsqueeze(1).expand(M, M, -1)              # (M, M, N)
        q = H_t.unsqueeze(0).expand(M, M, -1)              # (M, M, N)
        m = 0.5 * (p + q)

        kl_pm = (p * (p / (m + eps)).clamp(min=eps).log()).sum(dim=-1)  # (M, M)
        kl_qm = (q * (q / (m + eps)).clamp(min=eps).log()).sum(dim=-1)
        jsd = (0.5 * (kl_pm + kl_qm)).clamp(min=0.0)

        mu = jsd.mean(dim=1)
        mu_z = (mu - mu.mean()) / (mu.std() + eps)
        return torch.softmax(mu_z, dim=0)                  # (M,)


# ---------------------------------------------------------------------------
# Hybrid model — Cascaded-Parallel design
# ---------------------------------------------------------------------------

class HybridStockModel(nn.Module):
    """Cascaded-Parallel Hybrid THGNN × MaGNet stock prediction model.

    Combines:
    - MAGE temporal encoder (BiGRU-lite) for per-stock sequence modelling
    - TCH (MaGNet) for asynchronous cross-stock lead-lag causal discovery
    - Pos/Neg GAT (THGNN) for explicit synchronous correlation streams
    - GPH (MaGNet) for latent macro-thematic groupings
    - 4-stream semantic attention fusion + PairNorm (THGNN)
    - IC-ranked composite loss (external, in training script)

    Interface is drop-in compatible with StockHeteGAT:
        logits = model(features, pos_adj, neg_adj)  →  (N, 1)
    """

    def __init__(
        self,
        in_features: int = 12,
        embed_dim: int = 64,
        num_mage_layers: int = 1,
        num_moe_experts: int = 4,
        num_mha_heads: int = 2,
        gat_heads: int = 8,
        gat_out_features: int = 8,
        num_hyper_edges: int = 32,
        num_tch_hyper_edges: int = 32,
        num_tch_heads: int = 4,
        dropout: float = 0.1,
        predictor_out_dim: int = 1,
        predictor_activation: str | None = None,
    ):
        super().__init__()
        D = embed_dim

        # --- Input preprocessing ---
        self.input_norm = nn.LayerNorm(in_features)
        self.embed = nn.Linear(in_features, D)

        # --- MAGE temporal encoder ---
        self.mage_layers = nn.ModuleList([
            MAGEBlock(
                embed_dim=D,
                num_experts=num_moe_experts,
                num_heads=num_mha_heads,
                dropout=dropout,
            )
            for _ in range(num_mage_layers)
        ])

        self.drop = nn.Dropout(dropout)

        # --- Path A: Temporal-Causal Hypergraph (MaGNet) ---
        self.tch = TemporalCausalHypergraph(
            embed_dim=D,
            num_hyper_edges=num_tch_hyper_edges,
            num_heads=num_tch_heads,
            dropout=dropout,
        )

        # --- Path B: Pos / Neg GAT (THGNN) ---
        gat_dim = gat_heads * gat_out_features
        self.pos_gat = GraphAttnMultiHead(D, gat_out_features, num_heads=gat_heads)
        self.neg_gat = GraphAttnMultiHead(D, gat_out_features, num_heads=gat_heads)
        self.mlp_pos = nn.Linear(gat_dim, D)
        self.mlp_neg = nn.Linear(gat_dim, D)

        # --- Path B: Global Probabilistic Hypergraph (MaGNet) ---
        self.gph = GPHypergraph(embed_dim=D, num_hyper_edges=num_hyper_edges)

        # --- 4-stream semantic fusion + normalisation ---
        self.sem_attn = GraphAttnSemIndividual(in_features=D, hidden_size=D)
        self.pn = PairNorm(mode="PN-SI")

        # --- Prediction head ---
        layers: list[nn.Module] = [nn.Linear(D, predictor_out_dim)]
        if predictor_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        pos_adj: torch.Tensor,
        neg_adj: torch.Tensor,
        requires_weight: bool = False,
    ) -> torch.Tensor:
        # inputs: (N, T, F)

        # Preprocessing: normalise + cross-sectional demean
        x = self.input_norm(inputs)
        x = x - x.mean(dim=0, keepdim=True)

        # Feature embedding: (N, T, F) → (N, T, D)
        x = self.embed(x)

        # MAGE temporal encoding → Z_temp (N, T, D)
        z = x
        for mage in self.mage_layers:
            z = mage(z)

        # ── Path A: Temporal-Causal Hypergraph ──────────────────────────────
        # Takes the full 3D sequence; discovers asynchronous lead-lag causality
        h_causal = self.drop(self.tch(z))              # (N, D)

        # ── Path B: Cross-sectional streams on final-timestep state ─────────
        # The current state is pristine (un-mixed by TCH) → sharp cluster signals
        h_temp = self.drop(z[:, -1, :])                # (N, D)

        h_pos_raw, _ = self.pos_gat(h_temp, pos_adj)
        h_pos = self.drop(torch.relu(self.mlp_pos(h_pos_raw)))  # (N, D)

        h_neg_raw, _ = self.neg_gat(h_temp, neg_adj)
        h_neg = self.drop(torch.relu(self.mlp_neg(h_neg_raw)))  # (N, D)

        h_gph = self.gph(h_temp)                       # (N, D)

        # ── 4-stream semantic attention fusion ──────────────────────────────
        all_streams = torch.stack(
            [h_causal, h_pos, h_neg, h_gph], dim=1
        )                                               # (N, 4, D)
        fused, sem_weights = self.sem_attn(
            all_streams, requires_weight=requires_weight
        )                                               # (N, D)

        fused = self.pn(fused)
        out = self.predictor(fused)                    # (N, 1)

        if requires_weight:
            return out, sem_weights
        return out
