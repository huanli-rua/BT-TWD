"""强异质性合成数据集生成与加载入口。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils_logging import log_info


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _binary_search_bias(
    base_logit: np.ndarray,
    group_offsets: np.ndarray,
    eps: np.ndarray,
    target_rate: float,
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """通过二分搜索找到满足目标正例率的全局偏置。"""

    low, high = -8.0, 8.0
    bias = 0.0
    for _ in range(max_iter):
        bias = (low + high) / 2
        prob = _sigmoid(base_logit + group_offsets + bias + eps)
        rate = prob.mean()
        if abs(rate - target_rate) < tol:
            break
        if rate < target_rate:
            low = bias
        else:
            high = bias
    return bias


def _calc_group_stats(df: pd.DataFrame, group_col: str, target_col: str) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for g, sub in df.groupby(group_col):
        stats[str(g)] = {
            "count": int(len(sub)),
            "positive_rate": float(sub[target_col].mean()),
        }
    return stats


def generate_synth_strong_v1(
    n: int = 200_000,
    seed: int | None = 42,
    hetero_scale: float = 1.0,
    n_groups: int = 4,
    n_x: int = 10,
    n_z: int = 5,
    eps_std: float = 0.2,
    intercepts: Tuple[float, ...] | List[float] = (-2.0, -0.5, 0.5, 2.0),
    target_rate: float = 0.25,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    生成强异质性的二分类合成数据集。

    参数均使用中文日志描述，便于论文复现。
    """

    rng = np.random.default_rng(seed)
    if n_groups < 1:
        raise ValueError("n_groups 至少为 1")

    base_intercepts = np.array(intercepts, dtype=float)
    if n_groups > len(base_intercepts):
        # 若用户需要更多组别，使用线性插值扩展至指定组数
        base_intercepts = np.linspace(base_intercepts.min(), base_intercepts.max(), n_groups)
    else:
        base_intercepts = base_intercepts[:n_groups]

    group_labels = np.array([chr(ord("A") + i) for i in range(n_groups)], dtype=object)
    group_idx = rng.integers(0, n_groups, size=n)
    groups = group_labels[group_idx]
    scaled_intercepts = base_intercepts * hetero_scale
    group_offsets = np.take(scaled_intercepts, group_idx)

    x_features = rng.normal(0, 1, size=(n, n_x))
    z_features = rng.normal(0, 1, size=(n, n_z)) if n_z > 0 else None
    weights = rng.normal(0.0, 1.0, size=n_x)
    eps = rng.normal(0.0, eps_std, size=n)

    base_logit = x_features @ weights
    bias = _binary_search_bias(base_logit, group_offsets, eps, target_rate)
    logits = base_logit + group_offsets + bias + eps
    prob = _sigmoid(logits)
    expected_rate = float(prob.mean())
    y = rng.binomial(1, prob)

    data = {
        "target": y,
        "group": groups,
    }
    for i in range(n_x):
        data[f"x{i+1}"] = x_features[:, i]
    if n_z > 0 and z_features is not None:
        for j in range(n_z):
            data[f"z{j+1}"] = z_features[:, j]

    df = pd.DataFrame(data)
    group_stats = _calc_group_stats(df, "group", "target")
    pos_rate = float(df["target"].mean())

    meta = {
        "seed": seed,
        "n_samples": int(n),
        "n_groups": int(n_groups),
        "group_labels": group_labels.tolist(),
        "hetero_scale": float(hetero_scale),
        "base_intercepts": base_intercepts.tolist(),
        "scaled_intercepts": scaled_intercepts.tolist(),
        "n_x": int(n_x),
        "n_z": int(n_z),
        "eps_std": float(eps_std),
        "weights_mean": float(weights.mean()),
        "weights_std": float(weights.std()),
        "target_rate": pos_rate,
        "expected_rate": expected_rate,
        "target_rate_cfg": float(target_rate),
        "group_stats": group_stats,
        "bias": float(bias),
    }

    log_info(
        "【合成数据】生成完成："
        f"样本数={n}，分组={group_labels.tolist()}，全局正例率={pos_rate:.2%} (期望={expected_rate:.2%})，"
        "各组正例率如下"
    )
    for g in group_labels:
        stats = group_stats.get(g, {"count": 0, "positive_rate": float("nan")})
        log_info(
            f"组别 {g}: 样本数={stats['count']}，正例率={stats['positive_rate']:.2%}"
        )

    return df, meta


def save_synth_strong_v1(df: pd.DataFrame, meta: Dict[str, object], out_path: str, meta_path: str) -> None:
    out_file = Path(out_path)
    meta_file = Path(meta_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_file, index=False)
    with meta_file.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_info(
        f"【合成数据】数据已保存至 {out_file}，元数据写入 {meta_file}"
    )


def load_synth_strong_v1(path: str | Path) -> pd.DataFrame:
    """读取强异质性合成数据集，并打印基础统计。"""

    df = pd.read_csv(path)
    if "target" not in df.columns or "group" not in df.columns:
        raise KeyError("合成数据缺少 target 或 group 列，无法加载")
    pos_rate = df["target"].mean()
    log_info(
        f"【合成数据加载】文件={path}，样本数={len(df)}，全局正例率={pos_rate:.2%}"
    )
    for g, sub in df.groupby("group"):
        log_info(f"组别 {g}: 样本数={len(sub)}，正例率={sub['target'].mean():.2%}")
    return df


__all__ = [
    "generate_synth_strong_v1",
    "save_synth_strong_v1",
    "load_synth_strong_v1",
]
