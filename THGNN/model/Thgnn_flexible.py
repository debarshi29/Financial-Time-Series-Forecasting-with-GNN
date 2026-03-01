from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch
import math

class GraphAttnMultiHead(Module):
    def __init__(self, in_features, out_features, negative_slope=0.2, num_heads=4, bias=True, residual=True):
        super(GraphAttnMultiHead, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads*out_features)
        else:
            self.project = None
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2
        weight = self.leaky_relu(logits)
        masked_weight = torch.mul(weight, adj_mat).to_sparse()
        attn_weights = torch.sparse.softmax(masked_weight, dim=2).to_dense()
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)
        if self.bias is not None:
            support = support + self.bias
        if self.residual:
            support = support + self.project(inputs)
        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x
        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


class GraphAttnSemIndividual(Module):
    def __init__(self, in_features, hidden_size=128, act=nn.Tanh()):
        super(GraphAttnSemIndividual, self).__init__()
        self.project = nn.Sequential(nn.Linear(in_features, hidden_size),
                                     act,
                                     nn.Linear(hidden_size, 1, bias=False))

    def forward(self, inputs, requires_weight=False):
        w = self.project(inputs)
        beta = torch.softmax(w, dim=1)
        if requires_weight:
            return (beta * inputs).sum(1), beta
        else:
            return (beta * inputs).sum(1), None


class StockHeteGAT(nn.Module):
    """
    Flexible version of StockHeteGAT that automatically detects input feature dimensions.
    
    This version removes the hardcoded in_features=6 assumption and dynamically
    initializes the GRU based on the actual input shape during the first forward pass.
    """
    def __init__(
        self,
        in_features=6,  # Default, but will be overridden if input differs
        out_features=8,
        num_heads=8,
        hidden_dim=64,
        num_layers=1,
        predictor_out_dim=1,
        predictor_activation=None,
    ):
        super(StockHeteGAT, self).__init__()
        
        # Store initialization parameters
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_features = out_features
        self.num_heads = num_heads
        
        # Initialize GRU with default in_features
        # This will be re-initialized if needed on first forward pass
        self.encoding = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0.0  # Dropout only works with num_layers > 1
        )
        
        self.pos_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.neg_gat = GraphAttnMultiHead(
            in_features=hidden_dim,
            out_features=out_features,
            num_heads=num_heads
        )
        self.mlp_self = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_pos = nn.Linear(out_features*num_heads, hidden_dim)
        self.mlp_neg = nn.Linear(out_features*num_heads, hidden_dim)
        self.pn = PairNorm(mode='PN-SI')
        self.sem_gat = GraphAttnSemIndividual(in_features=hidden_dim,
                                              hidden_size=hidden_dim,
                                              act=nn.Tanh())
        predictor_layers = [nn.Linear(hidden_dim, predictor_out_dim)]
        if predictor_activation == 'sigmoid':
            predictor_layers.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*predictor_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
        
        # Flag to track if GRU has been initialized with correct input size
        self._gru_initialized = False

    def _reinitialize_gru_if_needed(self, input_features, device):
        """
        Reinitialize GRU if the input feature dimension doesn't match.
        
        Args:
            input_features: The actual number of features in the input data
            device: The device to move the new GRU to
        """
        if not self._gru_initialized or input_features != self.in_features:
            self.in_features = input_features
            
            # Create new GRU with correct input size
            self.encoding = nn.GRU(
                input_size=input_features,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=0.1 if self.num_layers > 1 else 0.0
            )
            
            # Move to correct device
            self.encoding = self.encoding.to(device)
            
            self._gru_initialized = True
            print(f"GRU re-initialized with input_size={input_features}, hidden_size={self.hidden_dim} on device {device}")

    def forward(self, inputs, pos_adj, neg_adj, requires_weight=False):
        # Check input shape and reinitialize GRU if needed
        # Expected input shape: (num_stocks, sequence_length, num_features)
        if inputs.dim() == 3:
            actual_features = inputs.size(-1)
            self._reinitialize_gru_if_needed(actual_features, inputs.device)
        elif inputs.dim() == 2:
            # If 2D input, assume it's already encoded
            # This shouldn't happen in normal operation but handle it gracefully
            print(f"Warning: Received 2D input with shape {inputs.shape}, expected 3D")
        
        # Process through GRU
        _, support = self.encoding(inputs)
        
        # Taking the last layer's hidden state
        # Shape: (num_layers, batch, hidden_dim) -> (batch, hidden_dim)
        support = support[-1]
        
        # DEBUG PRINTS
        # print(f"DEBUG: Support shape after GRU: {support.shape}")
        # print(f"DEBUG: Pos Adj shape: {pos_adj.shape}")
        # print(f"DEBUG: Neg Adj shape: {neg_adj.shape}")
        
        try:
            # Graph attention layers
            pos_support, pos_attn_weights = self.pos_gat(support, pos_adj, requires_weight)
            neg_support, neg_attn_weights = self.neg_gat(support, neg_adj, requires_weight)
        except Exception as e:
            print(f"ERROR in GAT layers: {e}")
            print(f"Support: {support.shape}")
            print(f"Pos Adj: {pos_adj.shape}")
            raise e
        
        try:
            # MLP transformations
            support = self.mlp_self(support)
            pos_support = self.mlp_pos(pos_support)
            neg_support = self.mlp_neg(neg_support)
            
            # Semantic attention
            all_embedding = torch.stack((support, pos_support, neg_support), dim=1)
            all_embedding, sem_attn_weights = self.sem_gat(all_embedding, requires_weight)
            all_embedding = self.pn(all_embedding)
            
            if requires_weight:
                return self.predictor(all_embedding), (pos_attn_weights, neg_attn_weights, sem_attn_weights)
            else:
                return self.predictor(all_embedding)
        except Exception as e:
            print(f"ERROR in post-GAT layers: {e}")
            try:
                print(f"Support shape: {support.shape}")
            except: pass
            raise e
