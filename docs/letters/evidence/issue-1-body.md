首先感谢weizheng的开源，不过我复现了你的实验，我发现你把测试集混进训练集，这么做的目的是啥？想问一下：

标题：该模型直接在测试集上进行训练——所有报告的指标均无效。

This repo's training loss includes the validation and test labels. The split exists in name only: train_mask is passed around but never connected to the loss. The model sees the exam answers during training. Everything below is verifiable by line number.
该代码库中的训练损失数据包含了验证集和测试集的标签。不过，这种划分只是名义上的而已： train_mask 这个参数虽然被传递给了模型，但实际上并未被用于计算损失值。在训练过程中，模型会看到所有的答案。所有内容都可以通过行号来验证。

Evidence:  证据：

train.py:17 — train_epoch(..., train_mask, ...) accepts the mask. The function body never uses it. Full labels goes into the model.
train.py:17 — train_epoch(..., train_mask, ...) 接受该掩码。该函数的主体部分从未使用过该掩码。完整的 labels 会被纳入模型中。
models/thg_oafn.py:152 —    models/thg_oafn.py:152 — —F.cross_entropy(logits, labels_processed) where labels_processed = all nodes, including test. Every gradient step optimizes on test answers. This is functionally identical to X_train = concat(X_train, X_test); y_train = concat(y_train, y_test).
thg_oafn.py:101-107 — GraphSMOTE picks minority nodes from all labels: test-set fraud nodes are oversampled into training.
thg_oafn.py:101-107 — GraphSMOTE 从所有类别中挑选出属于少数类的节点：测试集中的欺诈节点会被过度采样到训练数据中。
data_loader.py:257,281 — scaler fits the full dataset before splitting.
data_loader.py:257,281 — 在数据集被分割之前，缩放器会先对整个数据集进行缩放处理。
data_loader.py:265 — Amazon path passes no timestamps, so split_data splits by node order. The "temporal split verification" printout in train.py can never trigger on the flagship dataset. There is no temporal split.
data_loader.py:265 — Amazon 的路径处理方式不包含时间戳信息，因此 split_data 会按照节点的顺序来对数据进行分割。在 train.py 中出现的"时间分割验证"提示，实际上永远不会在旗舰数据集上出现。因为该数据集根本不存在时间分割的情况。
Measured impact (leak-free protocol, same trick isolated): appending evaluation rows to training takes AUPRC 0.6276 → 0.9340 (+0.31). That is the size of the artifact in the reported numbers.
经过测量，采用"无泄漏协议"且将评估数据单独处理后：在训练数据中添加评估行后，AUPRC 值从 0.6276 上升到了 0.9340，提升了 0.31。这一数值正是报告中所提到的"人为误差"的大小。

Conclusion: any metric produced by this pipeline measures memorization of the evaluation set, not fraud detection. Fix: connect
结论：该流程生成的任何指标都反映的是对评估数据的记忆情况，而非欺诈检测的结果。解决方案：进行相应的连接处理。train_mask to the loss, restrict oversampling to training nodes, fit the scaler on training rows.
