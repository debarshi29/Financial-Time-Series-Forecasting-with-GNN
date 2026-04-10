"""
Hybrid THGNN × MaGNet model.

Architecture (clean hybrid — no redundant components):
  Input (N, T, F)
    → LayerNorm + cross-sectional demean           [THGNN preprocessing]
    → Linear(F → D)                                [feature embedding]
    → MAGEBlock × num_mage_layers                  [MaGNet §3.3, BiGRU-lite]
        BiGRU(fwd+bwd) → Gating → SparseMoE → MHA
    → h = Z[:, -1, :]  ∈ (N, D)                   [final temporal state]
    → Pos GAT(h, pos_adj) → h_pos  ∈ (N, D)        [THGNN explicit correlation]
    → Neg GAT(h, neg_adj) → h_neg  ∈ (N, D)        [THGNN explicit correlation]
    → GPHypergraph(h)    → h_gph   ∈ (N, D)        [MaGNet §3.5.2 global groups]
    → 4-stream semantic attention fusion            [THGNN extended]
    → PairNorm-SI → Linear(D → 1)                  [THGNN]

Design rationale:
- TCH dropped: its causal attention over (T·N) is redundant with MAGE's temporal MHA.
- F.2D Attn dropped: MAGE's MHA already captures cross-timestep dependencies per stock.
- Pos/Neg GAT kept: provides explicit, pre-computed correlation-based inductive bias
  that complements GPH's learned probabilistic hyperedges.
- IC-ranked composite loss (MSE + Spearman IC + dispersion) kept from THGNN —
  better for cross-sectional portfolio ranking than MaGNet's BCE.
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
        weight = self.leaky_relu(f_1 + f_2)
        masked = torch.mul(weight, adj_mat).to_sparse()
        attn = torch.sparse.softmax(masked, dim=2).to_dense()
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
# New components from MaGNet
# ---------------------------------------------------------------------------

def _retanh(x: torch.Tensor) -> torch.Tensor:
    """ReTanh activation: 0 for x≤0, tanh(x) for x>0.
    Combines ReLU sparsity with tanh boundedness — keeps hyperedge
    assignments sparse while bounding them for numerical stability.
    """
    return torch.where(x <= 0, torch.zeros_like(x), torch.tanh(x))


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
        # Each expert: 2-layer FFN with GELU activation
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

        # Compute all experts: (K, D, E) — ensures every parameter gets gradients
        expert_outs = torch.stack(
            [expert(x_flat) for expert in self.experts], dim=-1
        )                                                       # (K, D, E)

        # Gather the assigned expert output per token
        idx = assignments.view(N * T, 1, 1).expand(N * T, D, 1)
        selected = expert_outs.gather(dim=2, index=idx).squeeze(2)  # (K, D)

        # Weight by routing probability (soft importance scaling)
        w = probs.gather(dim=1, index=assignments.unsqueeze(1))     # (K, 1)
        output = w * selected                                        # (K, D)

        return output.reshape(N, T, D)


class MAGEBlock(nn.Module):
    """Mamba-Attention-Gating-Experts block (MaGNet §3.3), BiGRU-lite variant.

    Replaces the Mamba SSM with a bidirectional GRU, eliminating the
    mamba-ssm dependency while preserving the same information-flow design:
      BiGRU(fwd + bwd) → Gating → SparseMoE → Multi-head self-attention

    All sub-layers use residual connections and LayerNorm for stable training.
    """

    def __init__(self, embed_dim: int, num_experts: int = 4,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        D = embed_dim

        # Bidirectional GRU: hidden_size=D per direction → output (N, T, 2D)
        self.bigru = nn.GRU(
            input_size=D, hidden_size=D,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        # Gating: learns soft weighting between forward and backward states
        # gate = σ(W_f · z_fwd + W_b · z_bwd)  — eq 5-6 in MaGNet
        self.gate_fwd = nn.Linear(D, D, bias=True)
        self.gate_bwd = nn.Linear(D, D, bias=False)

        self.norm_gru = nn.LayerNorm(D)

        # Sparse MoE for market-regime specialisation
        self.moe = SparseMoE(embed_dim=D, num_experts=num_experts, dropout=dropout)
        self.norm_moe = nn.LayerNorm(D)

        # Temporal self-attention (per stock, over T timesteps)
        # batch_first=True: input (N, T, D), output (N, T, D)
        self.mha = nn.MultiheadAttention(
            embed_dim=D, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_mha = nn.LayerNorm(D)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, D)

        # --- Bidirectional encoding + gating ---
        gru_out, _ = self.bigru(x)                         # (N, T, 2D)
        D = x.size(-1)
        z_fwd = gru_out[:, :, :D]                          # (N, T, D)
        z_bwd = gru_out[:, :, D:]                          # (N, T, D)

        gate = torch.sigmoid(
            self.gate_fwd(z_fwd) + self.gate_bwd(z_bwd)
        )                                                   # (N, T, D)
        z_G = gate * z_fwd + (1.0 - gate) * z_bwd         # (N, T, D)
        z_G = self.norm_gru(self.drop(z_G) + x)           # residual

        # --- Sparse MoE ---
        z_moe = self.moe(z_G)                              # (N, T, D)
        z_moe = self.norm_moe(self.drop(z_moe) + z_G)     # residual

        # --- Temporal self-attention ---
        z_attn, _ = self.mha(z_moe, z_moe, z_moe)        # (N, T, D)
        z_out = self.norm_mha(self.drop(z_attn) + z_moe)  # residual

        return z_out                                        # (N, T, D)


class GPHypergraph(nn.Module):
    """Global Probabilistic Hypergraph (MaGNet §3.5.2).

    Learns soft hyperedge assignments and weights each hyperedge by its
    uniqueness (Jensen-Shannon Divergence vs other hyperedges).
    Unique hyperedges capture distinct market sub-structures and receive
    higher weight in the final hypergraph convolution.

    Convolution: Z' = ELU(H · W · H^T · h · P) + h  (residual)
    where H ∈ (N, M) is the soft incidence matrix,
          W = diag(w_1, …, w_M) is the JSD-based importance weighting.
    """

    def __init__(self, embed_dim: int, num_hyper_edges: int = 32):
        super().__init__()
        D, M = embed_dim, num_hyper_edges
        # Project node features to hyperedge membership space
        self.to_hyper = nn.Linear(D, M)
        # Learnable node projection matrix P (eq 27 in MaGNet)
        self.proj = nn.Linear(D, D, bias=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, D)

        # Build soft incidence matrix
        H_raw = self.to_hyper(h)                    # (N, M)
        H_raw = _retanh(H_raw)                      # sparse + bounded
        H_GPH = torch.softmax(H_raw, dim=0)         # column-softmax: each column
                                                     # is a prob dist over N stocks

        # JSD-based hyperedge importance weights
        w = self._jsd_weights(H_GPH)                # (M,)
        W = torch.diag(w)                            # (M, M)

        # Hypergraph convolution
        HW = H_GPH @ W                              # (N, M)
        HWHT = HW @ H_GPH.t()                       # (N, N)
        z = F.elu(HWHT @ self.proj(h))              # (N, D)

        return self.norm(z + h)                     # residual connection

    @staticmethod
    def _jsd_weights(H_GPH: torch.Tensor) -> torch.Tensor:
        """Compute per-hyperedge importance via average Jensen-Shannon Divergence.

        Hyperedges that are more distinct from others (high avg JSD) capture
        unique market sub-structures and receive higher importance weight.
        Z-score normalisation + softmax ensures bounded, differentiable weights.
        """
        eps = 1e-8
        H_t = H_GPH.t()                                    # (M, N)
        M = H_t.size(0)

        # Pairwise mixture distributions m = 0.5*(p + q)
        p = H_t.unsqueeze(1).expand(M, M, -1)              # (M, M, N)
        q = H_t.unsqueeze(0).expand(M, M, -1)              # (M, M, N)
        m = 0.5 * (p + q)                                   # (M, M, N)

        # KL divergences: KL(p||m) and KL(q||m)
        kl_pm = (p * (p / (m + eps)).clamp(min=eps).log()).sum(dim=-1)  # (M, M)
        kl_qm = (q * (q / (m + eps)).clamp(min=eps).log()).sum(dim=-1)  # (M, M)
        jsd = (0.5 * (kl_pm + kl_qm)).clamp(min=0.0)       # (M, M), range [0, log2]

        mu = jsd.mean(dim=1)                                # (M,) avg JSD per hyperedge
        mu_z = (mu - mu.mean()) / (mu.std() + eps)          # z-score normalise
        return torch.softmax(mu_z, dim=0)                   # (M,) importance weights


# ---------------------------------------------------------------------------
# Hybrid model
# ---------------------------------------------------------------------------

class HybridStockModel(nn.Module):
    """Hybrid THGNN × MaGNet stock prediction model.

    Combines:
    - MAGE temporal encoder (BiGRU-lite) from MaGNet
    - Pos/Neg heterogeneous GAT from THGNN
    - Global Probabilistic Hypergraph from MaGNet
    - 4-stream semantic attention fusion + PairNorm from THGNN
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

        # --- Pos / Neg GAT (THGNN) ---
        gat_dim = gat_heads * gat_out_features
        self.pos_gat = GraphAttnMultiHead(D, gat_out_features, num_heads=gat_heads)
        self.neg_gat = GraphAttnMultiHead(D, gat_out_features, num_heads=gat_heads)

        # Project all streams to D
        self.mlp_self = nn.Linear(D, D)
        self.mlp_pos = nn.Linear(gat_dim, D)
        self.mlp_neg = nn.Linear(gat_dim, D)

        # --- Global Probabilistic Hypergraph (MaGNet) ---
        self.gph = GPHypergraph(embed_dim=D, num_hyper_edges=num_hyper_edges)

        # --- 4-stream semantic fusion + normalisation ---
        self.sem_attn = GraphAttnSemIndividual(in_features=D, hidden_size=D)
        self.pn = PairNorm(mode="PN-SI")

        # --- Prediction head ---
        layers: list[nn.Module] = [nn.Linear(D, predictor_out_dim)]
        if predictor_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*layers)

        # Weight initialisation
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
        x = x - x.mean(dim=0, keepdim=True)    # remove market factor

        # Feature embedding: (N, T, F) → (N, T, D)
        x = self.embed(x)

        # MAGE blocks: stacked temporal encoding
        z = x
        for mage in self.mage_layers:
            z = mage(z)                          # (N, T, D)

        # Extract final temporal state as node representation
        h = self.drop(z[:, -1, :])              # (N, D)

        # Stream 1: self projection
        h_self = self.drop(torch.relu(self.mlp_self(h)))      # (N, D)

        # Stream 2 & 3: pos/neg GAT over correlation graph
        h_pos_raw, _ = self.pos_gat(h, pos_adj)
        h_neg_raw, _ = self.neg_gat(h, neg_adj)
        h_pos = self.drop(torch.relu(self.mlp_pos(h_pos_raw)))  # (N, D)
        h_neg = self.drop(torch.relu(self.mlp_neg(h_neg_raw)))  # (N, D)

        # Stream 4: global probabilistic hypergraph
        h_gph = self.gph(h)                                     # (N, D)

        # 4-stream semantic attention fusion
        all_streams = torch.stack(
            [h_self, h_pos, h_neg, h_gph], dim=1
        )                                                        # (N, 4, D)
        fused, sem_weights = self.sem_attn(
            all_streams, requires_weight=requires_weight
        )                                                        # (N, D)

        fused = self.pn(fused)

        out = self.predictor(fused)                             # (N, 1)

        if requires_weight:
            return out, sem_weights
        return out
