# Code Evidence: five leakage locations in wei4zheng/THG-OAFN (branch: main)

All excerpts below are taken verbatim from the live repository at packaging time; full files are included alongside this document.

## train.py (lines 10-40)

```python
  10 | from tqdm import tqdm
  11 | 
  12 | from models.thg_oafn import create_thg_oafn_model
  13 | from utils.data_loader import load_synthetic_data, load_real_dataset, split_data
  14 | from utils.metrics import calculate_metrics, print_metrics
  15 | 
  16 | 
  17 | def train_epoch(model, g, features, labels, train_mask, optimizer, device):
  18 |     """Train for one epoch"""
  19 |     model.train()
  20 |     optimizer.zero_grad()
  21 | 
  22 |     # Move data to device
  23 |     features = features.to(device)
  24 |     labels = labels.to(device)
  25 |     g = g.to(device)
  26 | 
  27 |     # Forward propagation
  28 |     logits, embeddings, losses = model(
  29 |         g, features, labels, training=True
  30 |     )
  31 | 
  32 |     # Use model-internal loss (already considers oversampling)
  33 |     loss = losses['total_loss']
  34 | 
  35 |     # Backpropagation
  36 |     loss.backward()
  37 |     optimizer.step()
  38 | 
  39 |     return loss.item()
  40 | 
```

## train.py (lines 108-128)

```python
 108 | 
 109 |     # Verify temporal split (if timestamps exist)
 110 |     if dataset.timestamps is not None:
 111 |         import datetime
 112 |         train_timestamps = dataset.timestamps[train_mask]
 113 |         val_timestamps = dataset.timestamps[val_mask]
 114 |         test_timestamps = dataset.timestamps[test_mask]
 115 | 
 116 |         def ts_to_date(ts):
 117 |             """Convert timestamp to date string"""
 118 |             return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
 119 | 
 120 |         print(f"\nTemporal split verification:")
 121 |         print(f"  Train: {ts_to_date(train_timestamps.min())} to {ts_to_date(train_timestamps.max())}")
 122 |         print(f"  Val:   {ts_to_date(val_timestamps.min())} to {ts_to_date(val_timestamps.max())}")
 123 |         print(f"  Test:  {ts_to_date(test_timestamps.min())} to {ts_to_date(test_timestamps.max())}")
 124 |         print(f"  Train fraud ratio: {(labels[train_mask] == 1).sum().item() / train_mask.sum():.2%}")
 125 |         print(f"  Val fraud ratio:   {(labels[val_mask] == 1).sum().item() / val_mask.sum():.2%}")
 126 |         print(f"  Test fraud ratio:  {(labels[test_mask] == 1).sum().item() / test_mask.sum():.2%}")
 127 | 
 128 |     # Create model
```

## train.py (lines 148-160)

```python
 148 |     best_val_f1 = 0.0
 149 |     patience_counter = 0
 150 | 
 151 |     for epoch in range(args.epochs):
 152 |         # Training
 153 |         train_loss = train_epoch(
 154 |             model, g, features, labels,
 155 |             train_mask, optimizer, device
 156 |         )
 157 | 
 158 |         # Validation
 159 |         if (epoch + 1) % args.eval_every == 0:
 160 |             val_metrics = evaluate(model, g, features, labels, val_mask, device)
```

## thg_oafn.py (lines 95-160)

```python
  95 | 
  96 |         # Step 2: GRU-GNN temporal modeling
  97 |         h_temporal = self.gru_gnn(g, h_init)
  98 | 
  99 |         # Step 3: Oversampling (training only)
 100 |         graph_changed = False
 101 |         if training and labels is not None and self.oversample_ratio > 1.0:
 102 |             num_fraud = (labels == 1).sum().item()
 103 |             if num_fraud > 0:
 104 |                 h_oversampled, labels_oversampled, adj_oversampled = self.graph_smote(
 105 |                     h_temporal, labels,
 106 |                     adj_matrices[0] if adj_matrices else torch.eye(h_temporal.shape[0]).to(h_temporal.device),
 107 |                     oversample_ratio=self.oversample_ratio
 108 |                 )
 109 | 
 110 |                 if h_oversampled.shape[0] != h_temporal.shape[0]:
 111 |                     graph_changed = True
 112 |                     print(f"  Warning: Graph SMOTE changed node count from {h_temporal.shape[0]} to {h_oversampled.shape[0]}")
 113 |                     print(f"  Skipping graph-based layers for this batch")
 114 | 
 115 |                 h_processed = h_oversampled
 116 |                 labels_processed = labels_oversampled
 117 |             else:
 118 |                 print(f"  Warning: No fraud samples to oversample, skipping Graph SMOTE")
 119 |                 h_processed = h_temporal
 120 |                 labels_processed = labels
 121 |         else:
 122 |             h_processed = h_temporal
 123 |             labels_processed = labels
 124 | 
 125 |         # Step 4: Multi-layer attention fusion
 126 |         if graph_changed:
 127 |             h_attention = h_processed
 128 |         else:
 129 |             if adj_matrices is None:
 130 |                 num_nodes = h_processed.shape[0]
 131 |                 adj_matrices = None
 132 |             h_attention = self.multi_attention(g, h_processed, adj_matrices)
 133 | 
 134 |         # Step 5: Classification
 135 |         h_final = self.dropout(h_attention)
 136 |         logits = self.classifier(h_final)
 137 | 
 138 |         # Compute loss (for training)
 139 |         losses = {}
 140 |         if training and labels_processed is not None:
 141 |             num_fraud = (labels_processed == 1).sum().float()
 142 |             num_normal = (labels_processed == 0).sum().float()
 143 | 
 144 |             if num_fraud > 0 and num_normal > 0:
 145 |                 weight_ratio = (num_normal / num_fraud).sqrt()
 146 |                 weight_fraud = torch.clamp(weight_ratio, min=1.5, max=5.0).item()
 147 |                 class_weights = torch.tensor([1.0, weight_fraud]).to(logits.device)
 148 |                 print(f"  Class weights: Normal=1.0, Fraud={weight_fraud:.2f} (samples: {int(num_normal)}/{int(num_fraud)})")
 149 |             else:
 150 |                 class_weights = None
 151 | 
 152 |             loss_cls = F.cross_entropy(logits, labels_processed, weight=class_weights)
 153 |             losses['loss_cls'] = loss_cls
 154 |             losses['total_loss'] = loss_cls
 155 | 
 156 |         return logits, h_attention, losses if training else None
 157 | 
 158 |     def predict(self, g, node_features, adj_matrices=None):
 159 |         """
 160 |         Inference mode prediction
```

## data_loader.py (lines 134-186)

```python
 134 | def split_data(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
 135 |     """
 136 |     Split the dataset into train/val/test sets based on timestamps.
 137 | 
 138 |     Args:
 139 |         dataset: HeterogeneousGraphDataset object
 140 |         train_ratio: Train set ratio
 141 |         val_ratio: Validation set ratio
 142 |         test_ratio: Test set ratio
 143 | 
 144 |     Returns:
 145 |         train_mask, val_mask, test_mask: Boolean masks
 146 |     """
 147 |     num_nodes = dataset.num_nodes
 148 |     indices = np.arange(num_nodes)
 149 | 
 150 |     sorted_indices = np.argsort(dataset.timestamps) if dataset.timestamps is not None else indices
 151 |     train_size = int(num_nodes * train_ratio)
 152 |     val_size = int(num_nodes * val_ratio)
 153 | 
 154 |     train_indices = sorted_indices[:train_size]
 155 |     val_indices = sorted_indices[train_size:train_size + val_size]
 156 |     test_indices = sorted_indices[train_size + val_size:]
 157 | 
 158 |     train_mask = np.zeros(num_nodes, dtype=bool)
 159 |     val_mask = np.zeros(num_nodes, dtype=bool)
 160 |     test_mask = np.zeros(num_nodes, dtype=bool)
 161 |     train_mask[train_indices] = True
 162 |     val_mask[val_indices] = True
 163 |     test_mask[test_indices] = True
 164 | 
 165 |     return train_mask, val_mask, test_mask
 166 | 
 167 | 
 168 | def normalize_features(features, scaler=None):
 169 |     """
 170 |     Normalize node features.
 171 | 
 172 |     Args:
 173 |         features: Feature matrix
 174 |         scaler: Optional StandardScaler
 175 | 
 176 |     Returns:
 177 |         normalized_features: Scaled features
 178 |         scaler: Used scaler
 179 |     """
 180 |     if scaler is None:
 181 |         scaler = StandardScaler()
 182 |         normalized_features = scaler.fit_transform(features)
 183 |     else:
 184 |         normalized_features = scaler.transform(features)
 185 |     return normalized_features, scaler
 186 | 
```

## data_loader.py (lines 276-299)

```python
 276 |         raise ValueError("Node file must contain a 'label' column")
 277 | 
 278 |     labels = nodes_df['label'].values.astype(np.int64)
 279 |     feature_cols = [c for c in nodes_df.columns if c not in ['node_id', 'label']]
 280 |     node_features = nodes_df[feature_cols].values.astype(np.float32)
 281 |     node_features, _ = normalize_features(node_features)
 282 | 
 283 |     print(f"Loading edges from {edges_file}...")
 284 |     edges_df = pd.read_csv(edges_file, sep=separator)
 285 |     src, dst = edges_df['src'].values, edges_df['dst'].values
 286 |     edge_index = np.vstack([src, dst]).astype(np.int64)
 287 | 
 288 |     edge_types = edges_df['edge_type'].values.astype(np.int64) if 'edge_type' in edges_df.columns else np.zeros(len(edges_df), dtype=np.int64)
 289 |     timestamps = None
 290 |     if 'timestamp' in nodes_df.columns:
 291 |         timestamps = nodes_df['timestamp'].values
 292 |         timestamps = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min() + 1e-8)
 293 | 
 294 |     print(f"Dataset loaded: {len(labels)} nodes, {edge_index.shape[1]} edges")
 295 |     print(f"Features: {node_features.shape[1]} dimensions")
 296 |     print(f"Fraud ratio: {(labels == 1).sum() / len(labels):.2%}")
 297 | 
 298 |     dataset = HeterogeneousGraphDataset(node_features, edge_index, edge_types, labels, timestamps)
 299 |     return dataset
```

## Cited locations (verified against snapshot 6a3c1eb, author's last upload 2025-10-28; utils/data_loader.py unchanged since per GitHub commit history)

- train.py:17-39 — train_epoch accepts train_mask, body never uses it; full labels forwarded
- train.py:153-156 — training loop calls train_epoch with full labels every epoch
- thg_oafn.py:101-108 — GraphSMOTE samples minority nodes from ALL labels (test included)
- thg_oafn.py:152 — F.cross_entropy(logits, labels_processed) over all nodes (val + test included)
- utils/data_loader.py:281 with 168-185 — normalize_features (full-data fit_transform) called inside load_real_dataset BEFORE split_data
- utils/data_loader.py:289-292 — timestamps stays None unless node file has a timestamp column
- utils/data_loader.py:150 — split falls back to raw node order when timestamps is None
- train.py:109-126 — the 'Temporal split verification' block never executes when timestamps is None
