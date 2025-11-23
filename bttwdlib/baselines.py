import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .metrics import compute_binary_metrics, log_metrics
from .utils_logging import log_info


def _make_writable_matrix(X):
    """确保特征矩阵是可写的 numpy 数组。"""

    if sparse.issparse(X):
        # 对于随机森林，直接转换为稠密矩阵更稳妥
        X = X.toarray()
    else:
        X = np.asarray(X)

    if not X.flags.writeable:
        X = np.array(X, copy=True)
    return X


def _make_writable_vector(y):
    """确保标签向量是一维、可写的 numpy 数组。"""

    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    if not arr.flags.writeable:
        arr = np.array(arr, copy=True)
    return arr


def _aggregate_baseline_summary(per_fold_records: list[dict]) -> dict:
    """
    将基线模型的每折指标做均值/标准差汇总。
    per_fold_records: [{'Precision': ..., 'Recall': ..., ..., 'fold': 1}, ...]
    """
    if not per_fold_records:
        return {}

    # 取出所有列名，去掉 fold
    keys = set()
    for rec in per_fold_records:
        keys.update(rec.keys())
    keys.discard("fold")

    summary: dict = {}
    for col in sorted(keys):
        values = []
        for rec in per_fold_records:
            v = rec.get(col, np.nan)
            # 避免把 dict / list 之类塞进来，这里只聚合标量数值
            if isinstance(v, (int, float, np.number)) or v is None or np.isnan(v):
                values.append(v)
            else:
                # 如果真的有非数值（一般不会有），直接跳过该列
                values = None
                break

        if values is None:
            continue

        arr = np.array(values, dtype=float)
        summary[f"{col}_mean"] = float(np.nanmean(arr))
        summary[f"{col}_std"] = float(np.nanstd(arr))

    return summary



def _run_baseline_cv(model_builder, model_name: str, X, y, cfg, cv_splitter) -> dict:
    X = _make_writable_matrix(X)
    y = _make_writable_vector(y)

    costs = cfg.get("THRESHOLDS", {}).get("costs", {})
    metrics_cfg = cfg.get("METRICS", {})

    per_fold_records: list[dict] = []
    if isinstance(cv_splitter, StratifiedKFold):
        splitter = cv_splitter
    else:
        splitter = StratifiedKFold(
            n_splits=getattr(cv_splitter, "n_splits", 5),
            shuffle=True,
            random_state=42,
        )

    fold_idx = 1
    for train_idx, test_idx in splitter.split(X, y):
        clf = model_builder()
        clf.fit(X[train_idx], y[train_idx])

        y_pred = clf.predict(X[test_idx])
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X[test_idx])[:, 1]
        else:
            y_score = np.zeros_like(y_pred, dtype=float)

        metrics_dict = compute_binary_metrics(y[test_idx], y_pred, y_score, metrics_cfg, costs=costs)
        metrics_dict.setdefault("BND_ratio", 0.0)
        metrics_dict.setdefault("POS_Coverage", float("nan"))
        metrics_dict["fold"] = fold_idx
        per_fold_records.append(metrics_dict)
        fold_idx += 1

    summary = _aggregate_baseline_summary(per_fold_records)
    log_metrics(f"【基线-{model_name}】整体指标：", summary)
    return {"per_fold": per_fold_records, "summary": summary}


def train_eval_logreg(X, y, cfg, cv_splitter) -> dict:
    model_cfg = cfg.get("BASELINES", {}).get("logreg", {})

    def _builder():
        return LogisticRegression(max_iter=model_cfg.get("max_iter", 200), C=model_cfg.get("C", 1.0))

    return _run_baseline_cv(_builder, "LogReg", X, y, cfg, cv_splitter)


def train_eval_random_forest(X, y, cfg, cv_splitter) -> dict:
    rf_cfg = cfg.get("BASELINES", {}).get("random_forest", {})

    def _builder():
        return RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth"),
            random_state=rf_cfg.get("random_state", 42),
            n_jobs=cfg.get("EXP", {}).get("n_jobs", -1),
        )

    return _run_baseline_cv(_builder, "RF", X, y, cfg, cv_splitter)


def train_eval_knn(X, y, cfg, cv_splitter) -> dict:
    """
    使用 KNN 作为全局基线模型，进行 k 折交叉验证。
    """

    knn_cfg = cfg.get("BASELINES", {}).get("knn", {})

    def _builder():
        return KNeighborsClassifier(
            n_neighbors=knn_cfg.get("n_neighbors", 10),
        )

    return _run_baseline_cv(_builder, "KNN", X, y, cfg, cv_splitter)


def train_eval_xgboost(X, y, cfg, cv_splitter) -> dict:
    """
    使用 XGBoost 作为全局基线模型，进行 k 折交叉验证。
    """

    if not _XGB_AVAILABLE:
        raise RuntimeError("配置了 use_xgboost=True 但未安装 xgboost，请先安装该库。")

    xgb_cfg = cfg.get("BASELINES", {}).get("xgboost", {})

    def _builder():
        return XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 300),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
            random_state=xgb_cfg.get("random_state", 42),
            n_jobs=xgb_cfg.get("n_jobs", -1),
            eval_metric="logloss",
            use_label_encoder=False,
        )

    return _run_baseline_cv(_builder, "XGB", X, y, cfg, cv_splitter)
