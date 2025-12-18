
# bttwd_bucket_stats_clean.py
# ===========================
# 口径自检版 BT-TWD 桶级指标统计脚本
#
# Micro:
#   - 只看【强叶子桶】
#   - 按 n_val 加权
#
# Macro:
#   - 看【所有强桶】（可排除 ROOT）
#   - 不加权
#   - 输出 L1 / L2 / L3 / ALL
#
# Strong coverage:
#   = 强叶子桶样本数 / 全部叶子桶样本数
#
# Baseline 缺失：自动剔除

import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser("BT-TWD 桶级指标统计（口径自检版）")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--only-strong", action="store_true")
    parser.add_argument("--exclude-root", action="store_true")
    parser.add_argument("--weight-col", default="n_val")
    return parser.parse_args()

def mark_leaf(df):
    parents = set(df["parent_id"].dropna().unique())
    return ~df["bucket_id"].isin(parents)

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    need = ["bucket_id","parent_id","level","Regret","baseline_regret",args.weight_col,"is_weak"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"缺少列 {c}")

    df = df.copy()
    df["valid_pair"] = df["Regret"].notna() & df["baseline_regret"].notna()


    # 注意：coverage 的分母必须基于“全部叶子桶”，不应受 --only-strong 过滤影响
    df_all = df.copy()
    if args.only_strong:
        df = df[df["is_weak"] == False]

    df["is_leaf"] = mark_leaf(df)

    df_all["is_leaf"] = mark_leaf(df_all)
    # ---------- Micro（强叶子桶）----------
    # 分母：全部叶子桶（不受 --only-strong 影响）
    leaf_all = df_all[df_all["is_leaf"] & df_all["valid_pair"]]
    # 分子：强叶子桶
    leaf_strong = leaf_all[leaf_all["is_weak"] == False]

    w = args.weight_col
    total_leaf_w = leaf_all[w].sum()
    strong_leaf_w = leaf_strong[w].sum()

    strong_coverage = strong_leaf_w / total_leaf_w if total_leaf_w > 0 else np.nan

    if strong_leaf_w > 0:
        micro_win = (
            leaf_strong.loc[leaf_strong["Regret"] < leaf_strong["baseline_regret"], w].sum()
            / strong_leaf_w
        )
        micro_delta = (
            ((leaf_strong["baseline_regret"] - leaf_strong["Regret"]) * leaf_strong[w]).sum()
            / strong_leaf_w
        )
    else:
        micro_win = np.nan
        micro_delta = np.nan

    # ---------- Macro（所有强桶）----------
    macro_df = df[df["valid_pair"]]
    if args.exclude_root:
        macro_df = macro_df[macro_df["level"] > 0]

    macro_by_level = {}
    for lvl in sorted(macro_df["level"].unique()):
        sub = macro_df[macro_df["level"] == lvl]
        macro_by_level[f"L{lvl}"] = (sub["Regret"] < sub["baseline_regret"]).mean()

    macro_all = (macro_df["Regret"] < macro_df["baseline_regret"]).mean()

    # ---------- 输出 ----------
    print("========== Micro（强叶子桶，加权） ==========")
    print(f"micro win-rate (Regret): {micro_win:.6f}")
    print(f"micro ΔRegret: {micro_delta:.6f}")
    print(f"strong coverage: {strong_coverage:.6f}")
    print()
    print("========== Macro（所有强桶，不加权） ==========")
    for k,v in macro_by_level.items():
        print(f"{k}: {v:.6f}")
    print(f"ALL: {macro_all:.6f}")

if __name__ == "__main__":
    main()
