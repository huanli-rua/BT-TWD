from typing import Hashable, Optional, Set

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from .utils_logging import log_info
from .threshold_search import compute_regret


def predict_binary_by_cost(probs, costs: dict) -> np.ndarray:
    """根据成本矩阵与后验概率，选择期望损失更小的二分类预测标签。"""

    prob_arr = np.asarray(probs, dtype=float)
    p1 = prob_arr
    p0 = 1.0 - p1

    c_tp = costs.get("C_TP", 0.0)
    c_fp = costs.get("C_FP", 0.0)
    c_fn = costs.get("C_FN", 0.0)
    c_tn = costs.get("C_TN", 0.0)

    loss_pos = c_tp * p1 + c_fp * p0
    loss_neg = c_fn * p1 + c_tn * p0

    return np.where(loss_pos < loss_neg, 1, 0)


def compute_binary_metrics(y_true, y_pred, y_score, cfg_metrics, costs: dict | None = None) -> dict:
    pos_label = cfg_metrics.get("pos_label", 1)
    metrics_to_use = cfg_metrics.get(
        "use_metrics",
        ["Precision", "Recall", "F1", "BAC", "AUC", "MCC", "Kappa"],
    )
    output = {}
    if "Precision" in metrics_to_use:
        output["Precision"] = skm.precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if "Recall" in metrics_to_use:
        output["Recall"] = skm.recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if "F1" in metrics_to_use:
        output["F1"] = skm.f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    if "BAC" in metrics_to_use:
        output["BAC"] = skm.balanced_accuracy_score(y_true, y_pred)
    if "AUC" in metrics_to_use:
        try:
            output["AUC"] = skm.roc_auc_score(y_true, y_score)
        except Exception:
            output["AUC"] = np.nan
    if "MCC" in metrics_to_use:
        output["MCC"] = skm.matthews_corrcoef(y_true, y_pred)
    if "Kappa" in metrics_to_use:
        output["Kappa"] = skm.cohen_kappa_score(y_true, y_pred)
    metrics_to_use = cfg_metrics.get("use_metrics", [])
    if costs is not None and (not metrics_to_use or "Regret" in metrics_to_use):
        # Baseline 是二分类预测，直接把 y_pred 当作三支预测进行后悔值计算
        output["Regret"] = compute_regret(y_true, y_pred, costs)
        output.setdefault("BND_ratio", 0.0)
        output.setdefault("POS_Coverage", float("nan"))
    return output


def compute_s3_metrics(y_true, y_s3_pred, y_score, cfg_metrics, costs: dict | None = None) -> dict:
    """
    三支预测评估，将 BND 合并为负类计算常规指标，同时给出 BND 比例与 Regret。
    """

    y_s3_pred_arr = np.array(y_s3_pred)
    bnd_mask = (y_s3_pred_arr == -1) | (y_s3_pred_arr == "BND")
    bnd_ratio = bnd_mask.mean()
    pos_coverage = float(np.mean(y_s3_pred_arr == 1))

    if costs:
        y_pred_binary = predict_binary_by_cost(y_score, costs)
    else:
        y_pred_binary = np.where(y_s3_pred_arr == 1, 1, 0)

    metrics_dict = compute_binary_metrics(y_true, y_pred_binary, y_score, cfg_metrics)
    metrics_dict["BND_ratio"] = bnd_ratio
    metrics_dict["POS_Coverage"] = pos_coverage

    metrics_to_use = cfg_metrics.get("use_metrics", [])
    if costs is not None and (not metrics_to_use or "Regret" in metrics_to_use):
        metrics_dict["Regret"] = compute_regret(y_true, y_s3_pred_arr, costs)
    return metrics_dict


def evaluate_baseline_by_buckets(
    y_true,
    y_score,
    bucket_series,
    alpha,
    beta,
    cost_cfg,
    strong_buckets: Optional[Set[Hashable]] = None,
) -> list[dict]:
    """使用全局后验与阈值在每个桶上评估基线指标。

    若传入 ``strong_buckets``，仅对其中的强桶做全局阈值评估。
    """

    df = pd.DataFrame({"y_true": y_true, "y_score": y_score, "bucket_id": bucket_series})
    metrics_cfg = {"use_metrics": ["Precision", "Recall", "F1", "BAC", "AUC", "MCC", "Kappa"]}
    results = []

    for bucket_id, group in df.groupby("bucket_id"):
        if strong_buckets is not None and bucket_id not in strong_buckets:
            continue

        y_true_bucket = group["y_true"].to_numpy()
        y_score_bucket = group["y_score"].to_numpy()
        y_pred_s3 = np.where(y_score_bucket >= alpha, 1, np.where(y_score_bucket <= beta, 0, -1))
        y_pred_binary = np.where(y_score_bucket >= alpha, 1, 0)

        binary_metrics = compute_binary_metrics(y_true_bucket, y_pred_binary, y_score_bucket, metrics_cfg, costs=None)
        s3_metrics = compute_s3_metrics(y_true_bucket, y_pred_s3, y_score_bucket, metrics_cfg, costs=None)
        regret_val = compute_regret(y_true_bucket, y_pred_s3, cost_cfg)

        bucket_metrics = {
            "bucket_id": bucket_id,
            **binary_metrics,
            "Regret": regret_val,
            "BND_ratio": s3_metrics.get("BND_ratio"),
            "POS_Coverage": s3_metrics.get("POS_Coverage"),
            "n_samples": len(group),
            "alpha": alpha,
            "beta": beta,
        }
        results.append(bucket_metrics)

    return results


def log_metrics(prefix: str, metrics_dict: dict) -> None:
    items = ", ".join([f"{k}={v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else f"{k}={v}" for k, v in metrics_dict.items()])
    log_info(f"{prefix}{items}")
