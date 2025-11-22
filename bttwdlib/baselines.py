import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
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


def train_eval_logreg(X, y, cfg, cv_splitter) -> dict:
    # Ensure writable inputs for consistent downstream behavior
    X = _make_writable_matrix(X)
    y = _make_writable_vector(y)
    model_cfg = cfg.get("BASELINES", {}).get("logreg", {})
    clf = LogisticRegression(max_iter=model_cfg.get("max_iter", 200), C=model_cfg.get("C", 1.0))
    y_pred = cross_val_predict(clf, X, y, cv=cv_splitter, method="predict")
    y_score = cross_val_predict(clf, X, y, cv=cv_splitter, method="predict_proba")[:, 1]
    metrics_dict = compute_binary_metrics(y, y_pred, y_score, cfg.get("METRICS", {}))
    log_metrics("【基线-LogReg】整体指标：", metrics_dict)
    return {"per_fold": None, "summary": metrics_dict}


def train_eval_random_forest(X, y, cfg, cv_splitter) -> dict:
    # Ensure X, y are writable dense arrays for RandomForest
    X = _make_writable_matrix(X)
    y = _make_writable_vector(y)
    rf_cfg = cfg.get("BASELINES", {}).get("random_forest", {})
    clf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth"),
        random_state=rf_cfg.get("random_state", 42),
        n_jobs=cfg.get("EXP", {}).get("n_jobs", -1),
    )
    y_pred = cross_val_predict(clf, X, y, cv=cv_splitter, method="predict")
    y_score = cross_val_predict(clf, X, y, cv=cv_splitter, method="predict_proba")[:, 1]
    metrics_dict = compute_binary_metrics(y, y_pred, y_score, cfg.get("METRICS", {}))
    log_metrics("【基线-RF】整体指标：", metrics_dict)
    return {"per_fold": None, "summary": metrics_dict}
