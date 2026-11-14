# BT-TWD: Hierarchical Bucket-Tree Three-Way Decision

基于层次化桶树（Bucket-Tree）的三支决策（Three-Way Decision）分类框架：先按业务/统计规则将样本递归分桶，再在桶内搜索决策阈值并支持弱桶回退（fallback 到父桶 / ROOT），兼顾局部可靠性与整体性能。

## 仓库结构

```
bttwdlib/        核心库（模型、桶树构建、阈值搜索、交叉验证、基线、可视化）
configs/         8 个主实验数据集的 YAML 配置
data/            8 个公开数据集原始文件
notebooks/       8 个数据集的主实验 notebook
scripts/         辅助分析脚本
```

## 主实验数据集

| Notebook | 配置 | 数据集 |
|---|---|---|
| `01_adult_bttwd_main.ipynb` | `configs/adult_bttwd.yaml` | Adult Income |
| `03_bank_bttwd_main.ipynb` | `configs/bank_bttwd.yaml` | Bank Marketing |
| `06_online_shoppers_bttwd_main.ipynb` | `configs/online_shoppers.yaml` | Online Shoppers Intention |
| `07_telco_churn_bttwd_main.ipynb` | `configs/telco_churn.yaml` | Telco Customer Churn |
| `08_diabetic_bttwd_main.ipynb` | `configs/diabetic_bttwd.yaml` | Diabetes 130-US Hospitals |
| `09_hospital_bttwd_main.ipynb` | `configs/hospital_bttwd.yaml` | Hospital Readmissions |
| `10_weather_bttwd_main.ipynb` | `configs/weatherAUS_bttwd.yaml` | Rain in Australia |
| `11_credit_bttwd_main.ipynb` | `configs/credit_default_bttwd.yaml` | Default of Credit Card Clients |

## 快速开始

```bash
# 命令行运行单个数据集实验（分层切分 + K 折）
python bttwdlib/run_experiment.py --config configs/adult_bttwd.yaml
```

或直接运行对应 notebook。实验结果（指标、桶树结构、阈值等）写入本地 `results/<dataset>/`，该目录不入库，运行后自动生成。

## 环境

Python 3.11，依赖：numpy、pandas、scikit-learn、scipy、matplotlib、seaborn、pyyaml 等。
