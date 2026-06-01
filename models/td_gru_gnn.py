"""
Time-Decay GRU-GNN Fusion Module (TD-GRU-GNN)
Extends GRU-GNN with learnable temporal decay weighting.
Core formula: h_fused = alpha * (decay_weight * h_gru) + (1-alpha) * h_gnn
where decay_weight = exp(-lambda * delta_t)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


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


class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            return F.relu(self.W(h_neigh))


class TD_GRU_GNN(nn.Module):
    """
    Time-Decay GRU-GNN: applies exp(-lambda * delta_t) weighting to
    temporal features before fusing with structural features.

    When timestamps are None, degrades to standard weighted fusion.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, fusion_alpha=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_alpha = nn.Parameter(torch.tensor(fusion_alpha))
        self.decay_lambda = nn.Parameter(torch.tensor(0.1))

        self.gru = GRULayer(input_dim, hidden_dim)

        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, g, node_features, timestamps=None):
        """
        Args:
            g: DGL graph
            node_features: (N, input_dim)
            timestamps: (N,) normalized to [0,1], or None

        Returns:
            h_out: (N, output_dim)
        """
        num_nodes = node_features.shape[0]
        device = node_features.device

        h_gru = torch.zeros(num_nodes, self.hidden_dim, device=device)
        h_gru = self.gru(node_features, h_gru)

        # Time decay: recent transactions get higher weight
        if timestamps is not None:
            delta_t = 1.0 - timestamps.unsqueeze(1)  # (N, 1), newer = smaller delta
            decay_weight = torch.exp(-F.softplus(self.decay_lambda) * delta_t)
            h_gru_weighted = h_gru * decay_weight
        else:
            h_gru_weighted = h_gru

        # Structural modeling via GNN
        h_gnn = h_gru
        for layer in self.gnn_layers:
            h_gnn = layer(g, h_gnn)

        # Learnable fusion
        alpha = torch.sigmoid(self.fusion_alpha)
        h_fused = alpha * h_gru_weighted + (1 - alpha) * h_gnn

        return self.output_proj(h_fused)
