import numpy as np


def compute_bucket_score(metrics: dict, score_cfg: dict) -> float:
    """
    根据桶内评估指标计算桶的综合得分，分数越大越好。

    参数：
        metrics: dict，至少包含键：
            - "BAC" 或 "bac": 桶内 balanced accuracy，0~1
            - "Regret" 或 "regret": 桶内 regret（已按样本数归一化）
        score_cfg: dict，对应 cfg["SCORE"], 包含：
            - bucket_score_mode: "bac" / "regret" / "bac_regret"
            - bac_weight: float
            - regret_weight: float
            - regret_sign: float，一般是 -1.0
    """

    mode = str(score_cfg.get("bucket_score_mode", "bac_regret") or "").lower()
    w_bac = float(score_cfg.get("bac_weight", 1.0))
    w_reg = float(score_cfg.get("regret_weight", 1.0))
    reg_sign = float(score_cfg.get("regret_sign", -1.0))

    bac = metrics.get("BAC") if "BAC" in metrics else metrics.get("bac")
    regret = metrics.get("Regret") if "Regret" in metrics else metrics.get("regret")

    if mode == "bac":
        return -np.inf if bac is None else w_bac * float(bac)

    if mode == "regret":
        return -np.inf if regret is None else reg_sign * w_reg * float(regret)

    bac_val = 0.0 if bac is None else float(bac)
    regret_val = 0.0 if regret is None else float(regret)
    return w_bac * bac_val + reg_sign * w_reg * regret_val


def compute_bucket_gain(parent_score: float, child_scores: list[float], child_weights: list[float], gamma: float) -> float:
    """
    计算桶增益：Gain = sum(w_k * S_k) - S_parent - gamma * ΔN_bucket

    ΔN_bucket = 新增桶数（子桶个数-1），gamma 用于复杂度惩罚。
    """

    if len(child_scores) != len(child_weights):
        raise ValueError("child_scores 与 child_weights 长度不一致")

    weighted_child_score = float(np.sum(np.array(child_scores) * np.array(child_weights)))
    delta_bucket = max(len(child_scores) - 1, 0)
    return weighted_child_score - float(parent_score) - float(gamma) * delta_bucket
