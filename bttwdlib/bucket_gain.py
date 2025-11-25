import numpy as np

# 默认 score_metric="bac_regret" 等价于 score = BAC - Regret，可扩展为 BAC - λ*Regret 形式


def compute_bucket_score(metrics: dict, mode: str = "bac_regret") -> float:
    """
    输入桶的评估指标，输出一个用于结构决策的分数，分数越大越好。

    mode 支持：
    - "regret": 取负 regret（regret 越小越好，因此乘以 -1）
    - "bac": 直接使用平衡准确率 BAC
    - "bac_regret": BAC - regret（默认）
    """

    mode = str(mode or "").lower()
    regret = metrics.get("regret")
    bac = metrics.get("bac")

    if mode == "regret":
        return -np.inf if regret is None else -float(regret)
    if mode == "bac":
        return -np.inf if bac is None else float(bac)

    regret_val = 0.0 if regret is None else float(regret)
    bac_val = 0.0 if bac is None else float(bac)
    return bac_val - regret_val


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
