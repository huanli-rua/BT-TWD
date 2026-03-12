import pandas as pd
import numpy as np
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

    # 筛选弱叶子桶且测试样本数 > 0
    # 这里默认 bucket_metrics_gain_test_per_fold.csv 的每一行已经是叶子桶级别统计，
    # 因此只需按 is_weak == 1 过滤即可。
    weak_leaf_buckets = df[(df['is_weak'] == 1) & (df['n_test'] > 0)].copy()
    wbsr_dict = {'Dataset': dataset, 'Num_Weak_Leaf_Buckets': len(weak_leaf_buckets)}

    # 计算比例指标 WBSR
    for metric, higher_is_better in metrics_proportion.items():
        baseline_col = baseline_cols[metric]
        mask = (
            weak_leaf_buckets[[metric, baseline_col]].notna().all(axis=1)
            & np.isfinite(weak_leaf_buckets[[metric, baseline_col]]).all(axis=1)
        )
        weak_valid = weak_leaf_buckets[mask]

        if len(weak_valid) == 0:
            wbsr_ratio = np.nan
        else:
            if higher_is_better:
                wbsr_ratio = (weak_valid[metric] >= weak_valid[baseline_col]).mean()
            else:
                wbsr_ratio = (weak_valid[metric] <= weak_valid[baseline_col]).mean()
        wbsr_dict[f'{metric}_WBSR'] = wbsr_ratio

    # 计算分类性能增益
    for metric, higher_is_better in metrics_gain.items():
        baseline_col = baseline_cols[metric]
        mask = (
            weak_leaf_buckets[[metric, baseline_col]].notna().all(axis=1)
            & np.isfinite(weak_leaf_buckets[[metric, baseline_col]]).all(axis=1)
        )
        weak_valid = weak_leaf_buckets[mask]

        if len(weak_valid) == 0:
            avg_diff = np.nan
            weighted_avg_diff = np.nan
        else:
            if higher_is_better:
                diff = weak_valid[metric] - weak_valid[baseline_col]
            else:
                diff = weak_valid[baseline_col] - weak_valid[metric]
            avg_diff = diff.mean()
            weighted_avg_diff = (diff * weak_valid['n_test']).sum() / weak_valid['n_test'].sum()

        wbsr_dict[f'{metric}_Mean_Diff'] = avg_diff
        wbsr_dict[f'{metric}_Weighted_Mean_Diff'] = weighted_avg_diff

    summary_list.append(wbsr_dict)

# 汇总 DataFrame
wbsr_df = pd.DataFrame(summary_list).set_index('Dataset')
print('8个数据集弱叶子桶 BT-TWD 指标汇总（比例 + 分类性能增益）:')
print(wbsr_df)

# 保存结果 CSV
output_dir = Path('./wbsr_results')
output_dir.mkdir(exist_ok=True)
output_file = output_dir / 'wbsr_summary_weak_leaf_proportion_gain.csv'
wbsr_df.to_csv(output_file)
print(f'WBSR 汇总已保存到: {output_file}')
