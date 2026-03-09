# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, rankdata

# ===============================
# 文件路径
# ===============================
file_path = "bt vs baseline kfold.xlsx"

# ===============================
# 参数设置
# ===============================
metrics = ["Regret", "BAC"]
alpha = 0.05
control_method = "BTTWD"   # Bonferroni-Dunn 的控制算法
num_folds = 5  # 每个数据集的折数

# ===============================
# 读取指标数据
# ===============================
def load_metric(sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    datasets = df.columns[2:].tolist()  # 数据集列名
    methods = df['model'].unique().tolist()

    # 整理每折每算法的数据成 (N, k)
    values_list = []
    for m in methods:
        df_m = df[df['model'] == m]
        # flatten 所有折和数据集的指标
        values_list.append(df_m.iloc[:, 2:].values.flatten())

    # 转置得到 (N, k)
    values = np.array(values_list).T
    return datasets, methods, values

# ===============================
# 计算 rank（支持并列平均名次）
# ===============================
def compute_rank(values, higher_better=True):
    n, k = values.shape
    ranks = np.zeros((n, k), dtype=float)

    for i in range(n):
        row = values[i]
        if higher_better:
            ranks[i] = rankdata(-row, method="average")
        else:
            ranks[i] = rankdata(row, method="average")
    return ranks

# ===============================
# Friedman 检验
# ===============================
def friedman_test(ranks):
    n, k = ranks.shape
    mean_rank = ranks.mean(axis=0)
    chi2_f = (12 * n) / (k * (k + 1)) * np.sum(mean_rank**2) - 3 * n * (k + 1)
    p_value = 1 - chi2.cdf(chi2_f, df=k - 1)
    return mean_rank, chi2_f, p_value

# ===============================
# Bonferroni-Dunn 的 CD
# ===============================
def compute_bd_cd(k, n, alpha=0.05):
    z = norm.ppf(1 - alpha / (2 * (k - 1)))
    cd = z * math.sqrt(k * (k + 1) / (6 * n))
    return cd, z

# ===============================
# 画 Bonferroni-Dunn 图
# ===============================
def draw_bd_diagram(methods, mean_rank, cd, title, control_method):
    k = len(methods)
    order = np.argsort(mean_rank)
    methods_sorted = [methods[i] for i in order]
    mean_rank_sorted = mean_rank[order]

    if control_method not in methods:
        raise ValueError(f"控制算法 {control_method} 不在 methods 中")
    control_idx = methods.index(control_method)
    control_rank = mean_rank[control_idx]

    fig = plt.figure(figsize=(10, 3))
    ax = plt.gca()
    ax.set_xlim(k + 0.5, 0)
    ax.set_ylim(0, 1)
    ax.axis("off")

    y_axis = 0.8
    ax.plot([1, k], [y_axis, y_axis], lw=1, color="black")
    for r in range(1, k + 1):
        ax.plot([r, r], [y_axis, y_axis - 0.03], lw=1, color="black")
        ax.text(r, y_axis + 0.03, str(r), ha="center", va="bottom")

    y_cd = 0.95
    x_left = control_rank + cd
    x_right = control_rank
    x_left = min(x_left, k)

    ax.plot([x_right, x_left], [y_cd, y_cd], lw=2, color="black")
    ax.plot([x_right, x_right], [y_cd - 0.02, y_cd + 0.02], lw=2, color="black")
    ax.plot([x_left, x_left], [y_cd - 0.02, y_cd + 0.02], lw=2, color="black")
    ax.text((x_left + x_right) / 2, y_cd + 0.03, "CD", ha="center", va="bottom")

    for i, (m, r) in enumerate(zip(methods_sorted, mean_rank_sorted)):
        y_pos = 0.65 - i * 0.08
        text_x = 0.6
        gap = 0.05
        ax.plot([r, r], [y_axis, y_pos], lw=1, color="black")
        ax.plot([text_x + gap, r], [y_pos, y_pos], lw=1, color="black")
        ax.text(text_x, y_pos, m, ha="left", va="center")

    plt.title(title, pad=20)
    plt.savefig(title + ".pdf", bbox_inches="tight")
    plt.savefig(title + ".png", dpi=300, bbox_inches="tight")
    plt.show()

# ===============================
# 输出显著性比较结果
# ===============================
def report_significance(methods, mean_rank, cd, control_method):
    control_idx = methods.index(control_method)
    control_rank = mean_rank[control_idx]

    print(f"\n控制算法: {control_method}")
    print(f"控制算法平均排名: {control_rank:.4f}")
    print(f"Bonferroni-Dunn CD: {cd:.4f}\n")

    for m, r in zip(methods, mean_rank):
        if m == control_method:
            continue
        diff = r - control_rank
        if diff > cd:
            print(f"{control_method} 显著优于 {m}  (rank差={diff:.4f} > CD)")
        else:
            print(f"{control_method} 与 {m} 差异不显著  (rank差={diff:.4f} <= CD)")

# ===============================
# 主程序
# ===============================
title_map = {
    "Regret": "BT-TWD versus baseline algorithms comparison (Regret)",
    "BAC": "BT-TWD versus baseline algorithms comparison (BAC)"
}

for metric in metrics:
    datasets, methods, values = load_metric(metric)
    n, k = values.shape  # N = 折数 × 数据集数, k = 算法数

    if metric == "Regret":
        ranks = compute_rank(values, higher_better=False)
    else:
        ranks = compute_rank(values, higher_better=True)

    mean_rank, chi2_f, p_value = friedman_test(ranks)
    cd, z_value = compute_bd_cd(k, n, alpha=alpha)


    print(f"Bonferroni-Dunn critical value z (q_alpha) = {z_value:.4f}")
    print(f"Bonferroni-Dunn CD = {cd:.4f}")
    print("=" * 60)
    print(f"指标: {metric}")
    print(f"样本数 N = {n}, 算法数 k = {k}")
    print("平均排名:")
    for m, r in zip(methods, mean_rank):
        print(f"  {m}: {r:.4f}")

    print(f"\nFriedman 统计量 = {chi2_f:.4f}")
    print(f"Friedman p-value = {p_value:.6f}")
    print(f"Bonferroni-Dunn z = {z_value:.4f}")
    print(f"Bonferroni-Dunn CD = {cd:.4f}")

    report_significance(methods, mean_rank, cd, control_method)
    draw_bd_diagram(methods, mean_rank, cd, title_map[metric], control_method)
