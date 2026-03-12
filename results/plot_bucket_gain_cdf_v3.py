import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 配置区
# =========================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT_DIR, "cdf_results")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_FILES = {
    "Adult": os.path.join(ROOT_DIR, "adult_bttwd", "bucket_metrics_gain_test_per_fold.csv"),
    "Bank": os.path.join(ROOT_DIR, "bank_bttwd", "bucket_metrics_gain_test_per_fold.csv"),
    "Credit": os.path.join(ROOT_DIR, "credit_default_bttwd", "bucket_metrics_gain_test_per_fold.csv"),
    "Diabetic": os.path.join(ROOT_DIR, "diabetic_bttwd", "bucket_metrics_gain_test_per_fold.csv"),
    "Hospital": os.path.join(ROOT_DIR, "hospital_bttwd", "bucket_metrics_gain_test_per_fold.csv"),
    "Shopper": os.path.join(ROOT_DIR, "online_shoppers", "bucket_metrics_gain_test_per_fold.csv"),
    "Telco": os.path.join(ROOT_DIR, "telco_churn", "bucket_metrics_gain_test_per_fold.csv"),
    "WeatherAUS": os.path.join(ROOT_DIR, "weatherAUS_bttwd", "bucket_metrics_gain_test_per_fold.csv"),
}

# higher_better=True 表示越大越好；False 表示越小越好
METRICS = {
    "Regret": {"baseline_col": "baseline_regret", "higher_better": False},
    "F1": {"baseline_col": "baseline_f1", "higher_better": True},
    "BAC": {"baseline_col": "baseline_bac", "higher_better": True},
    "BND_ratio": {"baseline_col": "baseline_bnd_ratio", "higher_better": False},
    "POS_coverage": {"baseline_col": "baseline_pos_coverage", "higher_better": True},
}

# 是否额外输出单独的 dataset-wise 图和 pooled 图
DRAW_SEPARATE_FIGURES = True

# 图形风格参数
DATASET_LINE_ALPHA = 0.72
DATASET_LINE_WIDTH = 1.5
POOLED_LINE_WIDTH = 1.9
ZERO_LINE_WIDTH = 1.0
ZERO_LINE_ALPHA = 0.75
LEGEND_FONT_SIZE = 9
NOTE_FONT_SIZE = 10
TITLE_FONT_SIZE = 17
SUBTITLE_FONT_SIZE = 14


def ecdf(values: np.ndarray):
    """经验CDF"""
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def format_metric_display_name(metric_name: str) -> str:
    """图标题和坐标轴使用的指标显示名"""
    mapping = {
        "Regret": "regret",
        "F1": "F1",
        "BAC": "BAC",
        "BND_ratio": "BND ratio",
        "POS_coverage": "POS coverage",
    }
    return mapping.get(metric_name, metric_name)


def get_figure_title(metric_name: str) -> str:
    """生成总标题"""
    display_name = format_metric_display_name(metric_name)
    return f"CDF of bucket-level {display_name} improvement"


def get_x_label(metric_name: str, higher_better: bool) -> str:
    """生成横轴标签，统一使用 improvement 口径"""
    display_name = format_metric_display_name(metric_name)
    if higher_better:
        return f"{display_name} improvement (BT-TWD − baseline)"
    return f"{display_name} improvement (baseline − BT-TWD)"


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """
    读取并预处理数据：
    1. 仅保留 n_test > 0 的桶
    2. 保留 weak/strong 全部桶，因为弱桶只是阈值继承，并不是桶不存在
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在：{csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["n_test"]
    missing_base_cols = [c for c in required_cols if c not in df.columns]
    if missing_base_cols:
        raise ValueError(f"文件缺少必要列：{missing_base_cols} | 文件：{csv_path}")

    df = df[df["n_test"].fillna(0) > 0].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def compute_gain(df: pd.DataFrame, metric_name: str, baseline_col: str, higher_better: bool) -> pd.Series:
    """
    统一计算 improvement，使得 gain > 0 代表 BT-TWD 优于 baseline
    """
    required_cols = [metric_name, baseline_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"指标 {metric_name} 缺少列：{missing_cols}")

    sub = df[[metric_name, baseline_col]].dropna().copy()

    if higher_better:
        gain = sub[metric_name] - sub[baseline_col]
    else:
        gain = sub[baseline_col] - sub[metric_name]

    return gain


def save_summary_table(summary_rows: list, out_csv: str):
    """保存统计汇总表"""
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"已保存统计表：{out_csv}")


def collect_metric_data(metric_name: str, baseline_col: str, higher_better: bool):
    """
    收集某个指标的所有数据：
    1. 每个数据集的 gain 明细
    2. 汇总表
    3. pooled 明细
    """
    summary_rows = []
    dataset_gain_frames = []
    pooled_rows = []

    for dataset_name, csv_path in DATASET_FILES.items():
        try:
            df = load_and_prepare(csv_path)
            gain = compute_gain(df, metric_name, baseline_col, higher_better).dropna()

            if len(gain) == 0:
                print(f"[警告] {dataset_name} 的 {metric_name} 无有效数据，已跳过。")
                continue

            better_ratio = float((gain > 0).mean())
            non_worse_ratio = float((gain >= 0).mean())
            mean_gain = float(gain.mean())
            median_gain = float(gain.median())

            summary_rows.append({
                "dataset": dataset_name,
                "metric": metric_name,
                "n_buckets": int(len(gain)),
                "better_ratio": better_ratio,
                "non_worse_ratio": non_worse_ratio,
                "mean_gain": mean_gain,
                "median_gain": median_gain,
                "min_gain": float(gain.min()),
                "max_gain": float(gain.max()),
            })

            gain_df = pd.DataFrame({
                "dataset": dataset_name,
                "metric": metric_name,
                "gain": gain.to_numpy()
            })
            dataset_gain_frames.append(gain_df)
            pooled_rows.append(gain_df)

            print(
                f"{dataset_name} | {metric_name} | "
                f"桶数={len(gain)} | "
                f"BT优于比例={better_ratio:.4f} | "
                f"BT不劣于比例={non_worse_ratio:.4f} | "
                f"均值={mean_gain:.6f} | 中位数={median_gain:.6f}"
            )

        except Exception as e:
            print(f"[跳过] {dataset_name} 处理失败：{e}")

    if len(pooled_rows) == 0:
        empty = pd.DataFrame(columns=["dataset", "metric", "gain"])
        return summary_rows, empty, empty

    dataset_gain_df = pd.concat(dataset_gain_frames, ignore_index=True)
    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    return summary_rows, dataset_gain_df, pooled_df


def plot_zero_line(ax):
    """画更弱一点的 0 参考线"""
    ax.axvline(0, linestyle="--", linewidth=ZERO_LINE_WIDTH, alpha=ZERO_LINE_ALPHA)


def plot_dataset_cdf_on_ax(ax, metric_name: str, higher_better: bool, dataset_gain_df: pd.DataFrame):
    """在指定坐标轴上画 dataset-wise CDF"""
    for dataset_name in DATASET_FILES.keys():
        sub = dataset_gain_df[dataset_gain_df["dataset"] == dataset_name]
        if sub.empty:
            continue
        x, y = ecdf(sub["gain"].to_numpy())
        ax.step(
            x, y,
            where="post",
            label=dataset_name,
            alpha=DATASET_LINE_ALPHA,
            linewidth=DATASET_LINE_WIDTH,
        )

    plot_zero_line(ax)
    ax.set_xlabel(get_x_label(metric_name, higher_better))
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(a) Dataset-wise CDF", fontsize=SUBTITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)


def plot_pooled_cdf_on_ax(ax, metric_name: str, higher_better: bool, pooled_df: pd.DataFrame):
    """在指定坐标轴上画 pooled CDF"""
    pooled_gain = pooled_df["gain"].to_numpy()
    x, y = ecdf(pooled_gain)

    better_ratio = float((pooled_gain > 0).mean())
    non_worse_ratio = float((pooled_gain >= 0).mean())
    mean_gain = float(np.mean(pooled_gain))
    median_gain = float(np.median(pooled_gain))

    print(
        f"[Pooled] {metric_name} | "
        f"总桶数={len(pooled_gain)} | "
        f"BT优于比例={better_ratio:.4f} | "
        f"BT不劣于比例={non_worse_ratio:.4f} | "
        f"均值={mean_gain:.6f} | 中位数={median_gain:.6f}"
    )

    ax.step(x, y, where="post", label="All datasets", linewidth=POOLED_LINE_WIDTH)
    plot_zero_line(ax)
    ax.set_xlabel(get_x_label(metric_name, higher_better))
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(b) Pooled CDF", fontsize=SUBTITLE_FONT_SIZE)
    ax.legend(fontsize=LEGEND_FONT_SIZE)

    # 图中只保留最关键统计量，避免信息太挤。
    note = f"P(improvement > 0) = {better_ratio:.3f}"
    ax.text(
        0.97, 0.05, note,
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=NOTE_FONT_SIZE,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.82)
    )


def plot_metric_cdf_all_datasets(metric_name: str, higher_better: bool, dataset_gain_df: pd.DataFrame):
    """单独输出 dataset-wise CDF"""
    plt.figure(figsize=(8.5, 6.2))
    ax = plt.gca()
    plot_dataset_cdf_on_ax(ax, metric_name, higher_better, dataset_gain_df)
    plt.title(f"{format_metric_display_name(metric_name)} improvement (dataset-wise CDF)")
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, f"cdf_{metric_name.lower()}_all_datasets.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"已保存图像：{fig_path}")


def plot_metric_cdf_pooled(metric_name: str, higher_better: bool, pooled_df: pd.DataFrame):
    """单独输出 pooled CDF"""
    plt.figure(figsize=(8.2, 6.0))
    ax = plt.gca()
    plot_pooled_cdf_on_ax(ax, metric_name, higher_better, pooled_df)
    plt.title(f"{format_metric_display_name(metric_name)} improvement (pooled CDF)")
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, f"cdf_{metric_name.lower()}_pooled.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"已保存 pooled 图像：{fig_path}")


def plot_metric_cdf_combined(metric_name: str, higher_better: bool, dataset_gain_df: pd.DataFrame, pooled_df: pd.DataFrame):
    """输出拼图版：左边 dataset-wise，右边 pooled"""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))

    plot_dataset_cdf_on_ax(axes[0], metric_name, higher_better, dataset_gain_df)
    plot_pooled_cdf_on_ax(axes[1], metric_name, higher_better, pooled_df)

    fig.suptitle(get_figure_title(metric_name), fontsize=TITLE_FONT_SIZE, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig_path = os.path.join(OUT_DIR, f"cdf_{metric_name.lower()}_combined.pdf")
    plt.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"已保存拼图版图像：{fig_path}")


def save_metric_outputs(metric_name: str, summary_rows: list, dataset_gain_df: pd.DataFrame, pooled_df: pd.DataFrame):
    """保存汇总表和明细"""
    summary_csv = os.path.join(OUT_DIR, f"cdf_{metric_name.lower()}_summary.csv")
    save_summary_table(summary_rows, summary_csv)

    gain_detail_csv = os.path.join(OUT_DIR, f"cdf_{metric_name.lower()}_gain_details.csv")
    dataset_gain_df.to_csv(gain_detail_csv, index=False, encoding="utf-8-sig")
    print(f"已保存逐桶gain明细：{gain_detail_csv}")

    pooled_csv = os.path.join(OUT_DIR, f"cdf_{metric_name.lower()}_pooled_gain_details.csv")
    pooled_df.to_csv(pooled_csv, index=False, encoding="utf-8-sig")
    print(f"已保存 pooled 明细：{pooled_csv}")


def main():
    print("开始绘制 CDF 图……")
    print(f"根目录：{ROOT_DIR}")
    print(f"输出目录：{OUT_DIR}")
    print()

    print("检查数据文件：")
    for dataset_name, csv_path in DATASET_FILES.items():
        exists = os.path.exists(csv_path)
        print(f"{dataset_name:<12} | {'存在' if exists else '不存在'} | {csv_path}")
    print("=" * 80)

    for metric_name, cfg in METRICS.items():
        print(f"正在处理指标：{metric_name}")

        summary_rows, dataset_gain_df, pooled_df = collect_metric_data(
            metric_name=metric_name,
            baseline_col=cfg["baseline_col"],
            higher_better=cfg["higher_better"]
        )

        if dataset_gain_df.empty or pooled_df.empty:
            print(f"[警告] {metric_name} 无可用数据，已跳过。")
            print("-" * 80)
            continue

        save_metric_outputs(metric_name, summary_rows, dataset_gain_df, pooled_df)

        if DRAW_SEPARATE_FIGURES:
            plot_metric_cdf_all_datasets(
                metric_name=metric_name,
                higher_better=cfg["higher_better"],
                dataset_gain_df=dataset_gain_df
            )
            plot_metric_cdf_pooled(
                metric_name=metric_name,
                higher_better=cfg["higher_better"],
                pooled_df=pooled_df
            )

        plot_metric_cdf_combined(
            metric_name=metric_name,
            higher_better=cfg["higher_better"],
            dataset_gain_df=dataset_gain_df,
            pooled_df=pooled_df
        )

        print("=" * 80)

    print("全部完成。")


if __name__ == "__main__":
    main()
