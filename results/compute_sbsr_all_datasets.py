import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 数据集文件夹列表
dataset_dirs = [
    'adult_bttwd', 'bank_bttwd', 'credit_default_bttwd', 'diabetic_bttwd',
    'hospital_bttwd', 'online_shoppers', 'telco_churn', 'weatherAUS_bttwd'
]

# 指标及方向(True = 越高越好, False = 越低越好)
metrics_proportion = {
    'Regret': False,
    'BND_ratio': False,
    'POS_coverage': True
}

metrics_gain = {
    'BAC': True,
    'F1': True
}

baseline_cols = {
    'Regret': 'baseline_regret',
    'BND_ratio': 'baseline_bnd_ratio',
    'POS_coverage': 'baseline_pos_coverage',
    'BAC': 'baseline_bac',
    'F1': 'baseline_f1'
}

summary_list = []

for dataset in dataset_dirs:
    file_path = Path(dataset) / 'bucket_metrics_gain_test_per_fold.csv'
    df = pd.read_csv(file_path)
    
    # 筛选强桶且测试样本数 > 0
    strong_buckets = df[(df['is_weak'] == 0) & (df['n_test'] > 0)].copy()
    sbsr_dict = {'Dataset': dataset, 'Num_Strong_Buckets': len(strong_buckets)}
    
    # 计算比例指标 SBSR
    for metric, higher_is_better in metrics_proportion.items():
        baseline_col = baseline_cols[metric]
        mask = strong_buckets[[metric, baseline_col]].notna().all(axis=1) & np.isfinite(strong_buckets[[metric, baseline_col]]).all(axis=1)
        strong_valid = strong_buckets[mask]
        
        if len(strong_valid) == 0:
            sbsr_ratio = np.nan
        else:
            if higher_is_better:
                sbsr_ratio = (strong_valid[metric] >= strong_valid[baseline_col]).mean()
            else:
                sbsr_ratio = (strong_valid[metric] <= strong_valid[baseline_col]).mean()
        sbsr_dict[f'{metric}_SBSR'] = sbsr_ratio
    
    # 计算分类性能增益
    for metric, higher_is_better in metrics_gain.items():
        baseline_col = baseline_cols[metric]
        mask = strong_buckets[[metric, baseline_col]].notna().all(axis=1) & np.isfinite(strong_buckets[[metric, baseline_col]]).all(axis=1)
        strong_valid = strong_buckets[mask]
        
        if len(strong_valid) == 0:
            avg_diff = np.nan
            weighted_avg_diff = np.nan
        else:
            if higher_is_better:
                diff = strong_valid[metric] - strong_valid[baseline_col]
            else:
                diff = strong_valid[baseline_col] - strong_valid[metric]
            avg_diff = diff.mean()
            weighted_avg_diff = (diff * strong_valid['n_test']).sum() / strong_valid['n_test'].sum()
        sbsr_dict[f'{metric}_Mean_Diff'] = avg_diff
        sbsr_dict[f'{metric}_Weighted_Mean_Diff'] = weighted_avg_diff
    
    summary_list.append(sbsr_dict)

# 汇总 DataFrame
sbsr_df = pd.DataFrame(summary_list).set_index('Dataset')
print("8个数据集强桶 BT-TWD 指标汇总（比例 + 分类性能增益）:")
print(sbsr_df)

# 保存结果 CSV
output_dir = Path('./sbsr_results')
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'sbsr_summary_proportion_gain.csv'
sbsr_df.to_csv(output_file)
print(f"SBSR 汇总已保存到: {output_file}")