# BT-TWD: Hierarchical Bucket-Tree Three-Way Decision

基于层次化桶树（Bucket-Tree）的三支决策（Three-Way Decision）分类框架：先按业务/统计规则将样本递归分桶，再在桶内搜索决策阈值并支持弱桶回退（fallback 到父桶 / ROOT），兼顾局部可靠性与整体性能。

## 仓库结构

```
bttwdlib/        核心库（桶树构建、阈值搜索、三支决策模型、K折交叉验证、指标、t-SNE 可视化）
configs/         8 个主实验数据集的 YAML 配置
data/            8 个公开数据集原始文件
scripts/         辅助分析脚本
```

## 主实验数据集（configs/）

| 配置 | 数据集 |
|---|---|
| `adult_bttwd.yaml` | Adult Income |
| `bank_bttwd.yaml` | Bank Marketing |
| `online_shoppers.yaml` | Online Shoppers Intention |
| `telco_churn.yaml` | Telco Customer Churn |
| `diabetic_bttwd.yaml` | Diabetes 130-US Hospitals |
| `hospital_bttwd.yaml` | Hospital Readmissions |
| `weatherAUS_bttwd.yaml` | Rain in Australia |
| `credit_default_bttwd.yaml` | Default of Credit Card Clients |

## 快速开始

```bash
# 运行单个数据集主实验（分层切分 + K 折交叉验证）
python bttwdlib/run_experiment.py --config configs/adult_bttwd.yaml
```

实验结果（指标、桶树结构、阈值、逐样本预测等）写入本地 `results/<dataset>/`，该目录不入库，运行后自动生成。

## 环境

Python 3.11，依赖：numpy、pandas、scikit-learn、scipy、psutil、matplotlib、seaborn、pyyaml 等。
