# BT-TWD
Bucket Tree-TWD

## Baseline 阈值与桶级指标说明
- `baseline_analyzer.py` 中的基线只训练一个全局 XGBoost 模型，并在验证集搜索一对全局阈值 α/β。
- 评估阶段，baseline 仅对 BTTWD 过程中标记为强桶的叶子节点进行评估（当前实现中“强桶”定义为 `is_leaf == True` 且 `is_weak == False` 的桶），并在这些强桶上应用同一对全局阈值计算 BAC、Precision、Recall、Regret 等指标。弱桶不会单独计算 baseline 指标。
- 桶级基线指标的实现位于 `evaluate_baseline_by_buckets`，通过按照 `bucket_id` 分组并对每个强桶分组应用全局阈值来得到每个桶的指标。
