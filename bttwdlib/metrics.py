import numpy as np
from sklearn import metrics as skm
from .utils_logging import log_info


def compute_binary_metrics(y_true, y_pred, y_score, cfg_metrics) -> dict:
    pos_label = cfg_metrics.get("pos_label", 1)
    output = {}
    output["Precision"] = skm.precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    output["Recall"] = skm.recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    output["F1"] = skm.f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    output["BAC"] = skm.balanced_accuracy_score(y_true, y_pred)
    try:
        output["AUC"] = skm.roc_auc_score(y_true, y_score)
    except Exception:
        output["AUC"] = np.nan
    output["MCC"] = skm.matthews_corrcoef(y_true, y_pred)
    output["Kappa"] = skm.cohen_kappa_score(y_true, y_pred)
    return output


def compute_s3_metrics(y_true, y_s3_pred, y_score, cfg_metrics) -> dict:
    """
    三支预测评估，将 BND 合并为负类计算常规指标，同时给出 BND 比例。
    """
    y_s3_pred_arr = np.array(y_s3_pred)
    bnd_mask = (y_s3_pred_arr == -1) | (y_s3_pred_arr == "BND")
    y_pred_binary = np.where(y_s3_pred_arr == 1, 1, 0)
    bnd_ratio = bnd_mask.mean()
    metrics_dict = compute_binary_metrics(y_true, y_pred_binary, y_score, cfg_metrics)
    metrics_dict["BND_ratio"] = bnd_ratio
    return metrics_dict


def log_metrics(prefix: str, metrics_dict: dict) -> None:
    items = ", ".join([f"{k}={v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else f"{k}={v}" for k, v in metrics_dict.items()])
    log_info(f"{prefix}{items}")
