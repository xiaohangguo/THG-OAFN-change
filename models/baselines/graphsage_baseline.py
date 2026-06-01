import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv


class GraphSAGEBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim, 'mean'))
        for _ in range(num_layers - 1):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = F.relu(layer(g, h))
            h = self.dropout(h)
        return self.classifier(h)

    def predict(self, g, features):
        self.eval()
        with torch.no_grad():
            logits = self.forward(g, features)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs
