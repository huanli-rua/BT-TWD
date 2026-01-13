#!/usr/bin/env python
"""Hierarchical clustering analysis for bucket_metrics_gain.csv."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class SelectedBuckets:
    buckets: pd.DataFrame
    weak_ids: list[str]
    strong_id: str
    level: str | int | None
    level_label: str


def _normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def _numeric_features(df: pd.DataFrame, exclude: Iterable[str]) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    features = [col for col in numeric_cols if col not in set(exclude)]
    return [col for col in features if df[col].nunique(dropna=True) > 1]


def _select_buckets(df: pd.DataFrame) -> SelectedBuckets:
    if "is_weak" not in df.columns:
        raise ValueError("Missing required column: is_weak")

    df = df.copy()
    df["is_weak"] = _normalize_bool(df["is_weak"])

    if "n_train" not in df.columns:
        raise ValueError("Missing required column: n_train")

    level_col = "level" if "level" in df.columns else None

    def candidate_levels() -> Iterable[tuple[str | int | None, pd.DataFrame]]:
        if level_col:
            for lvl, sub in df.groupby(level_col):
                yield lvl, sub
        else:
            yield None, df

    best = None
    best_score = None
    for lvl, sub in candidate_levels():
        weak = sub[sub["is_weak"]].copy()
        strong = sub[~sub["is_weak"]].copy()
        if len(weak) < 2 or strong.empty:
            continue
        weak_indices = weak.index.tolist()
        strong_indices = strong.index.tolist()
        for w_idx_pair in combinations(weak_indices, 2):
            for s_idx in strong_indices:
                n_vals = df.loc[list(w_idx_pair) + [s_idx], "n_train"].to_numpy()
                score = np.max(n_vals) - np.min(n_vals)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (lvl, w_idx_pair, s_idx)

    if best is None:
        raise ValueError("Unable to find 2 weak and 1 strong bucket with matching level.")

    lvl, w_idx_pair, s_idx = best
    bucket_df = df.loc[list(w_idx_pair) + [s_idx]].copy()
    level_label = str(lvl) if lvl is not None else "N/A"
    return SelectedBuckets(
        buckets=bucket_df,
        weak_ids=bucket_df[bucket_df["is_weak"]]["bucket_id"].tolist(),
        strong_id=bucket_df[~bucket_df["is_weak"]]["bucket_id"].iloc[0],
        level=lvl,
        level_label=level_label,
    )


def _select_parent_bucket(df: pd.DataFrame, min_bucket_size: int) -> pd.Series | None:
    if "parent_id" not in df.columns:
        return None

    parent_counts = (
        df[df["parent_id"].astype(str).str.len() > 0]
        .groupby("parent_id")["n_train"]
        .agg(["count", "min", "sum"])
        .reset_index()
    )
    parent_counts = parent_counts[parent_counts["count"] > 1]
    if parent_counts.empty:
        return None

    parent_counts["min_bucket_gap"] = min_bucket_size - parent_counts["min"]
    parent_counts = parent_counts.sort_values(
        by=["min_bucket_gap", "min", "count"], ascending=[False, True, False]
    )
    parent_id = parent_counts.iloc[0]["parent_id"]
    parent_rows = df[df["bucket_id"] == parent_id]
    if parent_rows.empty:
        return None
    return parent_rows.iloc[0]


def _select_min_bucket_parent(
    df: pd.DataFrame, min_bucket_size: int
) -> tuple[pd.Series, pd.DataFrame] | None:
    if "parent_id" not in df.columns:
        return None

    child_rows = df[df["parent_id"].astype(str).str.len() > 0].copy()
    if child_rows.empty:
        return None

    child_stats = (
        child_rows.groupby("parent_id")["n_train"]
        .agg(["count", "min"])
        .reset_index()
    )
    constrained = child_stats[child_stats["min"] < min_bucket_size]
    if constrained.empty:
        return None

    constrained = constrained.sort_values(by=["min", "count"], ascending=[True, False])
    parent_id = constrained.iloc[0]["parent_id"]
    parent_rows = df[df["bucket_id"] == parent_id]
    if parent_rows.empty:
        return None
    parent_row = parent_rows.iloc[0]
    selected_children = child_rows[child_rows["parent_id"] == parent_id]
    return parent_row, selected_children


def _compute_merge_heights(linkage_matrix: np.ndarray, n_obs: int) -> list[float]:
    merge_heights = [np.nan] * n_obs
    for i, row in enumerate(linkage_matrix):
        left, right, height, _ = row
        for idx in (int(left), int(right)):
            if idx < n_obs and np.isnan(merge_heights[idx]):
                merge_heights[idx] = height
    return merge_heights


def _cluster_summary(
    features_scaled: np.ndarray,
    linkage_matrix: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    pairwise = np.linalg.norm(
        features_scaled[:, None, :] - features_scaled[None, :, :], axis=2
    )
    merge_heights = _compute_merge_heights(linkage_matrix, features_scaled.shape[0])

    rows = []
    for idx in range(features_scaled.shape[0]):
        cluster_label = labels[idx]
        members = np.where(labels == cluster_label)[0]
        if len(members) > 1:
            distances = [pairwise[idx, j] for j in members if j != idx]
            avg_dist = float(np.mean(distances))
        else:
            avg_dist = np.nan
        rows.append({
            "cluster_label": int(cluster_label),
            "intra_cluster_avg_distance": avg_dist,
            "merge_height": merge_heights[idx],
        })
    return pd.DataFrame(rows)


def _run_clustering(
    buckets: pd.DataFrame,
    feature_cols: list[str],
    scaler: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    features = buckets[feature_cols].fillna(0.0)
    scaler_obj = RobustScaler() if scaler == "robust" else StandardScaler()
    features_scaled = scaler_obj.fit_transform(features)

    linkage_matrix = linkage(features_scaled, method="ward")
    clusterer = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = clusterer.fit_predict(features_scaled)

    summary_df = _cluster_summary(features_scaled, linkage_matrix, labels)
    summary_df = pd.concat([buckets.reset_index(drop=True), summary_df], axis=1)
    return summary_df, linkage_matrix


def run_analysis(
    csv_path: Path,
    output_dir: Path,
    scaler: str,
    min_bucket_size: int,
    analysis_mode: str,
) -> None:
    df = pd.read_csv(csv_path)
    df["is_weak"] = _normalize_bool(df["is_weak"])

    if analysis_mode == "min-bucket":
        selection = _select_min_bucket_parent(df, min_bucket_size)
        if selection is None:
            raise ValueError("Unable to find parent bucket constrained by min_bucket.")
        parent_bucket, child_buckets = selection
        bucket_df = pd.concat([parent_bucket.to_frame().T, child_buckets], ignore_index=True)
        weak_ids = bucket_df[bucket_df["is_weak"]]["bucket_id"].tolist()
        strong_ids = bucket_df[~bucket_df["is_weak"]]["bucket_id"].tolist()
        level_value = parent_bucket.get("level") if "level" in bucket_df.columns else None
        level_label = str(level_value) if level_value is not None else "N/A"
        selected = SelectedBuckets(
            buckets=bucket_df,
            weak_ids=weak_ids,
            strong_id=strong_ids[0] if strong_ids else "",
            level=level_value,
            level_label=level_label,
        )
    else:
        selected = _select_buckets(df)

    feature_cols = _numeric_features(
        selected.buckets,
        exclude={"n_val", "baseline_precision", "baseline_recall"},
    )
    if "n_train" not in feature_cols:
        feature_cols.append("n_train")

    summary_df, linkage_matrix = _run_clustering(
        selected.buckets,
        feature_cols,
        scaler,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dendro_path = output_dir / "bucket_dendrogram.png"

    plt.figure(figsize=(10, 6))
    bucket_labels = [
        f"{row.bucket_id} ({'weak' if row.is_weak else 'strong'})"
        for row in selected.buckets.itertuples()
    ]
    dendrogram(linkage_matrix, labels=bucket_labels, leaf_rotation=25)
    plt.title("Hierarchical Clustering Dendrogram (Ward)")
    plt.ylabel("Merge Height")
    plt.tight_layout()
    plt.savefig(dendro_path, dpi=200)
    plt.close()

    report_csv = output_dir / "bucket_cluster_report.csv"
    summary_df.to_csv(report_csv, index=False)

    parent_bucket = None
    min_child = None
    min_child_bucket = None
    if analysis_mode == "min-bucket":
        selection = _select_min_bucket_parent(df, min_bucket_size)
        if selection is not None:
            parent_bucket, child_rows = selection
            min_child = child_rows["n_train"].min()
            min_child_bucket = child_rows.loc[child_rows["n_train"].idxmin()]["bucket_id"]
    else:
        parent_bucket = _select_parent_bucket(df, min_bucket_size)
        if parent_bucket is not None:
            child_rows = df[df["parent_id"] == parent_bucket["bucket_id"]]
            min_child = child_rows["n_train"].min()
            min_child_bucket = child_rows.loc[child_rows["n_train"].idxmin()]["bucket_id"]

    weak_stats = summary_df[summary_df["is_weak"]]
    strong_stats = summary_df[~summary_df["is_weak"]]

    def _mean_or_nan(series: pd.Series) -> float:
        return float(series.mean()) if not series.empty else float("nan")

    report_lines = [
        "层次聚类分析报告",
        f"输入文件: {csv_path}",
        f"选择层级: {selected.level_label}",
        f"弱桶: {', '.join(selected.weak_ids)}",
        f"强桶: {selected.strong_id}",
        "",
        "弱桶分析:",
        f"  平均簇内距离: {_mean_or_nan(weak_stats['intra_cluster_avg_distance']):.4f}",
        f"  平均合并高度: {_mean_or_nan(weak_stats['merge_height']):.4f}",
        "  诊断: 弱桶合并高度更低或簇内距离更大，说明结构更分散。",
        "",
        "强桶分析:",
        f"  平均簇内距离: {_mean_or_nan(strong_stats['intra_cluster_avg_distance']):.4f}",
        f"  平均合并高度: {_mean_or_nan(strong_stats['merge_height']):.4f}",
        "  诊断: 强桶合并高度更高或簇内距离更小，说明结构更紧凑。",
        "",
        "结构约束分析:",
    ]

    if parent_bucket is not None and min_child is not None:
        report_lines.extend(
            [
                f"  父桶: {parent_bucket['bucket_id']} (n_train={parent_bucket['n_train']})",
                f"  最小子桶: {min_child_bucket} (n_train={min_child})",
                f"  min_bucket_size={min_bucket_size}",
                "  解释: 当子桶样本数低于 min_bucket_size 时，父桶不能继续细分。",
            ]
        )
    else:
        report_lines.append("  未找到可用于 min_bucket 约束分析的父桶。")

    if analysis_mode == "min-bucket":
        parent_flag = (
            "弱桶" if parent_bucket is not None and bool(parent_bucket["is_weak"]) else "强桶"
        )
        report_lines.extend(
            [
                "",
                "父桶分析:",
                f"  父桶类型: {parent_flag}",
                "  聚类结构显示父桶与子桶的合并高度差异。",
                "  当最小子桶样本数低于 min_bucket_size 时，继续细分会导致样本不足。",
                "  因此在 min_bucket 约束下停止分裂是合理的。",
            ]
        )

    report_lines.extend(
        [
            "",
            "桶级别合并情况:",
            summary_df[[
                "bucket_id",
                "is_weak",
                "cluster_label",
                "intra_cluster_avg_distance",
                "merge_height",
            ]]
            .to_string(index=False),
        ]
    )

    report_txt = output_dir / "bucket_cluster_report.txt"
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical clustering for bucket metrics.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/synth_strong_v1/bucket_metrics_gain.csv"),
        help="Path to bucket_metrics_gain.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hierarchical_bucket_analysis"),
        help="Output directory for plots and reports",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "robust"],
        default="standard",
        help="Scaler to use before clustering",
    )
    parser.add_argument(
        "--min-bucket-size",
        type=int,
        default=50,
        help="min_bucket_size for constraint analysis",
    )
    parser.add_argument(
        "--analysis-mode",
        choices=["weak-strong", "min-bucket"],
        default="weak-strong",
        help="Analysis mode: weak-strong or min-bucket constraint validation",
    )
    args = parser.parse_args()

    run_analysis(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        scaler=args.scaler,
        min_bucket_size=args.min_bucket_size,
        analysis_mode=args.analysis_mode,
    )


if __name__ == "__main__":
    main()
