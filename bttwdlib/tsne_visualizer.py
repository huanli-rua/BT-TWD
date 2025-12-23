from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from .bucket_rules import BucketTree
from .bttwd_model import BTTWDModel
from .config_loader import load_yaml_cfg
from .data_loader import load_dataset
from .preprocessing import prepare_features_and_labels
from .utils_logging import log_info
from .utils_seed import set_global_seed


def _is_xgb_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def _ensure_estimators(cfg: dict, force_logreg_global: bool) -> None:
    """确保 BTTWD 配置中使用的估计器可用，必要时自动降级为 logreg。"""

    bcfg = cfg.setdefault("BTTWD", {})
    xgb_available = _is_xgb_available()

    global_estimator = str(bcfg.get("global_estimator", "logreg")).lower()
    if force_logreg_global or global_estimator in {"", "none", "null", "disabled"}:
        bcfg["global_estimator"] = "logreg"
    elif global_estimator in {"xgb", "xgboost"} and not xgb_available:
        log_info("【t-SNE】未检测到 xgboost，global_estimator 自动回退到 logreg")
        bcfg["global_estimator"] = "logreg"

    bucket_estimator = str(bcfg.get("bucket_estimator", bcfg.get("posterior_estimator", "logreg"))).lower()
    if bucket_estimator in {"xgb", "xgboost"} and not xgb_available:
        log_info("【t-SNE】未检测到 xgboost，bucket_estimator 自动回退到 logreg")
        bcfg["bucket_estimator"] = "logreg"
    elif bucket_estimator in {"", "none", "null", "disabled"}:
        # 保持显式禁用的配置，t-SNE 可视化仍可运行
        bcfg["bucket_estimator"] = "none"


def _prepare_dataset(
    cfg: dict, sample_size: int | None, random_state: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, BucketTree]:
    """读取配置并完成特征工程，返回子样本及桶树对象。"""

    df_raw, target_col = load_dataset(cfg)
    log_info(f"【t-SNE】数据加载完成：样本数={len(df_raw)}，目标列={target_col}")

    X, y, meta = prepare_features_and_labels(df_raw, cfg)

    prep_cfg = cfg.get("PREPROCESS", {})
    bucket_cols: list[str] = (prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or [])
    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    for lvl in bucket_levels:
        col_name = lvl.get("col") or lvl.get("feature")
        if col_name and col_name not in bucket_cols:
            bucket_cols.append(col_name)

    df_processed = meta.get("df_processed", df_raw)
    bucket_df = df_processed[bucket_cols].reset_index(drop=True)

    if sample_size is not None and len(y) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(y), size=sample_size, replace=False)
        X = X[indices]
        y = y[indices]
        bucket_df = bucket_df.iloc[indices].reset_index(drop=True)
        log_info(f"【t-SNE】已按 sample_size={sample_size} 子采样，剩余样本数={len(y)}")

    bucket_tree = BucketTree(bucket_levels, feature_names=bucket_cols)
    return X, y, bucket_df, bucket_tree


def _compute_tsne_embedding(
    X: np.ndarray, perplexity: float, learning_rate: float, random_state: int
) -> np.ndarray:
    """对输入特征执行 t-SNE，并适配可行的 perplexity。"""

    n_samples = len(X)
    max_perplexity = max(5.0, (n_samples - 1) / 3)
    effective_perplexity = min(perplexity, max_perplexity)
    if effective_perplexity < 1:
        effective_perplexity = 5.0

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
        init="random",
    )
    embedding = tsne.fit_transform(X)
    log_info(
        f"【t-SNE】嵌入完成：n_samples={n_samples}，perplexity={effective_perplexity:.1f}，learning_rate={learning_rate}"
    )
    return embedding


def _build_effective_bucket_map(model: BTTWDModel, bucket_ids: Iterable[str]) -> dict[str, str]:
    """基于模型阈值回退结果，为每个桶ID生成有效桶映射。"""

    mapping: dict[str, str] = {}
    unique_ids = pd.unique(pd.Series(list(bucket_ids)))
    for bid in unique_ids:
        _, source = model._get_threshold_with_backoff(str(bid))
        mapping[str(bid)] = source
    return mapping


def _collect_mode_result(
    cfg: dict,
    mode_label: str,
    bucket_tree: BucketTree,
    X: np.ndarray,
    y: np.ndarray,
    bucket_df: pd.DataFrame,
    embedding: np.ndarray,
    output_root: Path,
) -> dict:
    """训练/推理指定模式（开启/关闭回退），并打包 t-SNE 可视化所需的数据。"""

    cfg_mode = deepcopy(cfg)
    cfg_mode.setdefault("BTTWD", {})
    cfg_mode["BTTWD"]["use_gain_weak_backoff"] = mode_label == "fallback_on"

    mode_dir = output_root / f"{mode_label}_artifacts"
    cfg_mode.setdefault("OUTPUT", {})
    cfg_mode["OUTPUT"]["results_dir"] = str(mode_dir)

    mode_display = "回退开启" if mode_label == "fallback_on" else "回退关闭"
    log_info(f"【t-SNE】开始训练模式：{mode_display}")

    bucket_tree_mode = BucketTree(bucket_tree.levels_cfg, feature_names=bucket_tree.feature_names)
    model = BTTWDModel.from_cfg(
        cfg_mode, feature_names=bucket_df.columns.tolist(), bucket_tree=bucket_tree_mode
    )
    model.fit(X, y, bucket_df)

    proba = model.predict_proba(X, bucket_df)
    preds = model.predict(X, bucket_df)
    assigned_buckets = model.bucket_tree.assign_buckets(bucket_df)
    effective_map = _build_effective_bucket_map(model, assigned_buckets)
    effective_bucket_ids = assigned_buckets.map(lambda bid: effective_map.get(str(bid), "ROOT"))
    used_fallback = effective_bucket_ids != assigned_buckets

    df_mode = pd.DataFrame(
        {
            "tsne_x": embedding[:, 0],
            "tsne_y": embedding[:, 1],
            "y_true": y,
            "y_pred_s3": preds,
            "y_score": proba,
            "bucket_id": assigned_buckets,
            "effective_bucket_id": effective_bucket_ids,
            "used_fallback": used_fallback.astype(int),
            "mode": mode_display,
            "mode_key": mode_label,
        }
    )
    csv_path = output_root / f"tsne_embedding_{mode_label}.csv"
    df_mode.to_csv(csv_path, index=False)
    log_info(f"【t-SNE】模式 {mode_display} 嵌入数据已写入：{csv_path}")

    fallback_stats_df = pd.DataFrame(model.fallback_stats.values()) if model.fallback_stats else pd.DataFrame()
    fallback_stats_path = output_root / f"fallback_stats_{mode_label}.csv"
    if not fallback_stats_df.empty:
        fallback_stats_df.to_csv(fallback_stats_path, index=False)

    summary = {
        "mode": mode_label,
        "mode_display": mode_display,
        "n_samples": len(df_mode),
        "fallback_samples": int(used_fallback.sum()),
        "fallback_rate": float(used_fallback.mean()) if len(df_mode) else float("nan"),
        "pos_rate": float(np.mean(y)) if len(y) else float("nan"),
        "pred_pos_rate": float(np.mean(preds == 1)) if len(df_mode) else float("nan"),
        "boundary_rate": float(np.mean(preds == -1)) if len(df_mode) else float("nan"),
        "csv_path": str(csv_path),
        "fallback_stats_path": str(fallback_stats_path) if not fallback_stats_df.empty else "",
    }

    return {"mode": mode_label, "mode_display": mode_display, "df": df_mode, "summary": summary}


def _plot_tsne_modes(results: list[dict], figure_path: Path) -> None:
    """对比不同回退模式的 t-SNE 分布图。"""

    n_modes = len(results)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), sharex=True, sharey=True)
    if n_modes == 1:
        axes = [axes]  # type: ignore[list-item]

    for ax, res in zip(axes, results):
        df = res["df"]
        scatter = ax.scatter(
            df["tsne_x"],
            df["tsne_y"],
            c=df["y_pred_s3"],
            cmap="coolwarm",
            alpha=0.7,
            s=12,
            edgecolors="none",
        )
        fallback_mask = df["used_fallback"] == 1
        if fallback_mask.any():
            ax.scatter(
                df.loc[fallback_mask, "tsne_x"],
                df.loc[fallback_mask, "tsne_y"],
                color="#f39c12",
                s=18,
                alpha=0.6,
                label="使用回退阈值/模型",
            )
            ax.legend(loc="best")

        ax.set_title(f"{res.get('mode_display', res.get('mode', ''))}\n(n={len(df)})")
        ax.set_xlabel("t-SNE X")
        ax.set_ylabel("t-SNE Y")

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.9)
    cbar.set_label("预测标签（-1 边界 / 0 阴性 / 1 阳性）")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    log_info(f"【t-SNE】对比图已保存：{figure_path}")


def visualize_fallback_with_tsne(
    config_path: str,
    output_dir: str = "results/tsne_fallback",
    sample_size: int | None = 2000,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    random_state: int = 42,
    force_logreg_global: bool = False,
) -> dict:
    # 加载配置文件
    cfg = load_yaml_cfg(config_path)
    set_global_seed(random_state)

    # 确保估计器的选择正确
    _ensure_estimators(cfg, force_logreg_global)

    # 准备数据集
    X, y, bucket_df, bucket_tree = _prepare_dataset(cfg, sample_size, random_state)

    # 计算 t-SNE 嵌入
    embedding = _compute_tsne_embedding(X, perplexity, learning_rate, random_state)

    # 创建输出目录
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 收集结果并保存
    results = []
    for mode_label in ("fallback_on", "fallback_off"):
        results.append(
            _collect_mode_result(cfg, mode_label, bucket_tree, X, y, bucket_df, embedding, output_root)
        )

    combined_df = pd.concat([res["df"] for res in results], ignore_index=True)
    combined_path = output_root / "tsne_fallback_embedding.csv"
    combined_df.to_csv(combined_path, index=False)
    log_info(f"【t-SNE】已保存 t-SNE 嵌入与模式标签：{combined_path}")

    summary_path = output_root / "tsne_fallback_summary.csv"
    summary_df = pd.DataFrame([res["summary"] for res in results])
    summary_df.to_csv(summary_path, index=False)

    # 输出图片
    figure_path = output_root / "tsne_fallback_compare.png"
    _plot_tsne_modes(results, figure_path)

    # 返回结果
    return {
        "embedding_path": combined_path,
        "figure_path": figure_path,
        "summary_path": summary_path,
        "results": results,
    }
