"""
GPU-compatible Time-Decay GRU-GNN Fusion Module.
Uses PyTorch sparse matrix ops instead of DGL for full CUDA support.
v2: Added residual connections and BatchNorm for better feature preservation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        r_t = torch.sigmoid(self.W_r(combined))
        z_t = torch.sigmoid(self.W_z(combined))
        combined_reset = torch.cat([x_t, r_t * h_prev], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t


class SparseGraphConvLayer(nn.Module):
    """GCN layer with residual connection (GraphSAGE-style concat)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.W_self = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, adj, h):
        h_agg = torch.sparse.mm(adj, h)
        h_out = self.W_neigh(h_agg) + self.W_self(h)
        h_out = self.bn(h_out)
        return F.relu(h_out)


class TD_GRU_GNN_GPU(nn.Module):
    """
    GPU-compatible Time-Decay GRU-GNN.
    v2: Residual GNN layers preserve self-features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, fusion_alpha=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_alpha = nn.Parameter(torch.tensor(fusion_alpha))
        self.decay_lambda = nn.Parameter(torch.tensor(0.1))

        self.gru = GRULayer(input_dim, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            SparseGraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, adj, node_features, timestamps=None):
        num_nodes = node_features.shape[0]
        device = node_features.device

        h_gru = torch.zeros(num_nodes, self.hidden_dim, device=device)
        h_gru = self.gru(node_features, h_gru)

        if timestamps is not None:
            delta_t = 1.0 - timestamps.unsqueeze(1)
            decay_weight = torch.exp(-F.softplus(self.decay_lambda) * delta_t)
            h_gru_weighted = h_gru * decay_weight
        else:
            h_gru_weighted = h_gru

        h_gnn = h_gru
        for layer in self.gnn_layers:
            h_gnn = layer(adj, h_gnn) + h_gnn

        alpha = torch.sigmoid(self.fusion_alpha)
        h_fused = alpha * h_gru_weighted + (1 - alpha) * h_gnn

        return self.output_proj(h_fused)
