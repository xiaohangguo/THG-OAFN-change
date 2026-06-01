import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv


class GATBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(input_dim, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout))
        self.classifier = nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h).flatten(1)
            h = F.elu(h)
        return self.classifier(h)

    def predict(self, g, features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
