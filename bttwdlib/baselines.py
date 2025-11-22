import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from .metrics import compute_binary_metrics, log_metrics
from .utils_logging import log_info


def _make_writable_matrix(X):
    """Return a writable copy of X without breaking sparse inputs."""

    if sparse.issparse(X):
        return X.copy()

    arr = np.asarray(X)
    if arr.flags.writeable:
        return arr
    return np.array(arr, copy=True)


def _make_writable_vector(y):
    """Return a writable 1D array for target labels."""

    arr = np.asarray(y)
    if arr.flags.writeable:
        return arr
    return np.array(arr, copy=True)


def train_eval_logreg(X, y, cfg, cv_splitter) -> dict:
    # Ensure writable inputs while preserving sparse matrices
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
    # Ensure writable inputs while preserving sparse matrices
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
