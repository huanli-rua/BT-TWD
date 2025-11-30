import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def compute_regret(y_true, y_pred_s3, costs: dict) -> float:
    """Compute average regret (cost) for three-way predictions.

    Args:
        y_true: Array-like of ground truth labels (0/1).
        y_pred_s3: Array-like of predictions encoded as 1 (POS), 0 (NEG), -1/"BND" (BND).
        costs: Dict containing regret costs for different outcomes.

    Returns:
        Average regret value.
    """

    y_true = np.asarray(y_true)
    pred_arr = np.asarray(y_pred_s3)

    pred_numeric = np.where(pred_arr == "BND", -1, pred_arr)

    cost = np.zeros_like(y_true, dtype=float)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask

    cost[pos_mask & (pred_numeric == 1)] = costs.get("C_TP", 0.0)
    cost[pos_mask & (pred_numeric == 0)] = costs.get("C_FN", 0.0)
    cost[pos_mask & (pred_numeric == -1)] = costs.get("C_BP", 0.0)

    cost[neg_mask & (pred_numeric == 1)] = costs.get("C_FP", 0.0)
    cost[neg_mask & (pred_numeric == 0)] = costs.get("C_TN", 0.0)
    cost[neg_mask & (pred_numeric == -1)] = costs.get("C_BN", 0.0)

    if len(cost) == 0:
        return float("nan")
    return float(cost.mean())


def search_thresholds_with_regret(
    prob: np.ndarray,
    y_true: np.ndarray,
    alpha_grid,
    beta_grid,
    costs: dict,
    gap_min: float = 0.0,
    tol: float = 1e-12,
):
    """Grid search (alpha, beta) by minimizing regret with tie-breaking.

    Decision rule:
        p >= alpha -> POS
        p <= beta -> NEG
        else       -> BND

    Args:
        prob: posterior probabilities for the positive class.
        y_true: ground truth binary labels.
        alpha_grid: iterable of alpha candidates.
        beta_grid: iterable of beta candidates.
        costs: regret cost matrix.
        gap_min: enforce alpha >= beta + gap_min.
        tol: tolerance for regret tie-breaking.

    Returns:
        best_alpha, best_beta, stats_dict
    """

    best_alpha = None
    best_beta = None
    best_stats = None

    auc_val = float("nan")
    try:
        if np.unique(y_true).size >= 2:
            auc_val = float(roc_auc_score(y_true, prob))
    except Exception:
        auc_val = float("nan")

    for alpha in alpha_grid:
        for beta in beta_grid:
            if alpha < beta + gap_min:
                continue

            preds = np.where(prob >= alpha, 1, np.where(prob <= beta, 0, -1))
            regret_val = compute_regret(y_true, preds, costs)
            pred_binary = np.where(preds == 1, 1, 0)

            precision = precision_score(y_true, pred_binary, zero_division=0)
            recall = recall_score(y_true, pred_binary, zero_division=0)
            f1 = f1_score(y_true, pred_binary, zero_division=0)
            bnd_ratio = float(np.mean(preds == -1))
            pos_coverage = float(np.mean(preds == 1))

            pos_mask = y_true == 1
            neg_mask = ~pos_mask
            tp = np.sum((preds == 1) & pos_mask)
            tn = np.sum((preds == 0) & neg_mask)
            tpr = tp / pos_mask.sum() if pos_mask.sum() > 0 else np.nan
            tnr = tn / neg_mask.sum() if neg_mask.sum() > 0 else np.nan
            if np.isnan(tpr) and np.isnan(tnr):
                bac = np.nan
            elif np.isnan(tpr):
                bac = tnr / 2
            elif np.isnan(tnr):
                bac = tpr / 2
            else:
                bac = 0.5 * (tpr + tnr)

            stats = {
                "regret": regret_val,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "bac": float(bac) if not np.isnan(bac) else np.nan,
                "bnd_ratio": bnd_ratio,
                "pos_coverage": pos_coverage,
                "n_samples": int(len(prob)),
                "auc": auc_val,
            }

            if best_stats is None:
                best_alpha, best_beta, best_stats = alpha, beta, stats
                continue

            if regret_val + tol < best_stats["regret"]:
                best_alpha, best_beta, best_stats = alpha, beta, stats
            elif abs(regret_val - best_stats["regret"]) <= tol:
                if f1 > best_stats["f1"] + tol:
                    best_alpha, best_beta, best_stats = alpha, beta, stats
                elif abs(f1 - best_stats["f1"]) <= tol and bnd_ratio < best_stats["bnd_ratio"] - tol:
                    best_alpha, best_beta, best_stats = alpha, beta, stats

    if best_alpha is None:
        best_alpha = float(alpha_grid[0]) if len(alpha_grid) else 0.5
        best_beta = float(beta_grid[0]) if len(beta_grid) else 0.0
        best_stats = {
            "regret": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "bnd_ratio": float("nan"),
            "pos_coverage": float("nan"),
            "n_samples": 0,
        }

    return best_alpha, best_beta, best_stats

