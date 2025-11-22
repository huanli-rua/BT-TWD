import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .utils_logging import log_info


def _infer_columns(df: pd.DataFrame, target_col: str):
    continuous_cols = df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.drop(columns=[target_col]).select_dtypes(exclude=["number"]).columns.tolist()
    return continuous_cols, categorical_cols


def prepare_features_and_labels(df: pd.DataFrame, cfg: dict):
    """
    返回 X, y, meta：
    - X: 编码后的特征矩阵
    - y: 0/1 标签
    - meta: 特征名等辅助信息
    """
    prep_cfg = cfg.get("PREPROCESS", {})
    data_cfg = cfg.get("DATA", {})
    target_col = data_cfg.get("target_col", "income")
    positive_label = data_cfg.get("positive_label", ">50K")

    df = df.copy()
    # 缺失值统一处理
    if prep_cfg.get("handle_missing") == "question_mark":
        df.replace("?", np.nan, inplace=True)

    # 推断列
    continuous_cols = prep_cfg.get("continuous_cols") or []
    categorical_cols = prep_cfg.get("categorical_cols") or []
    if not continuous_cols and not categorical_cols:
        continuous_cols, categorical_cols = _infer_columns(df, target_col)
    log_info(f"【预处理】连续特征={len(continuous_cols)}个，类别特征={len(categorical_cols)}个")

    y = (df[target_col] == positive_label).astype(int).values
    X_raw = df.drop(columns=[target_col])

    transformers = []
    if categorical_cols:
        encoder = OneHotEncoder(drop="first" if prep_cfg.get("drop_first") else None, handle_unknown="ignore")
        transformers.append(("cat", encoder, categorical_cols))
    if continuous_cols:
        scaler_type = prep_cfg.get("scaler_type", "standard")
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        transformers.append(("num", scaler, continuous_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(steps=[("preprocess", preprocessor)])
    X = pipeline.fit_transform(X_raw)

    # Ensure the transformed feature matrix is writable to avoid downstream
    # errors in cross-validation that expect a mutable buffer.
    if sparse.issparse(X):
        X = X.copy()
        if not X.data.flags.writeable:
            X.data = np.array(X.data, copy=True)
    else:
        X = np.asarray(X)
        if not X.flags.writeable:
            X = np.array(X, copy=True)

    # 生成特征名
    feature_names = []
    if categorical_cols:
        cat_encoder: OneHotEncoder = pipeline.named_steps["preprocess"].named_transformers_["cat"]
        feature_names.extend(cat_encoder.get_feature_names_out(categorical_cols).tolist())
    if continuous_cols:
        feature_names.extend(continuous_cols)

    log_info(f"【预处理】编码后维度={X.shape[1]}")

    meta = {
        "feature_names": feature_names,
        "continuous_cols": continuous_cols,
        "categorical_cols": categorical_cols,
        "preprocess_pipeline": pipeline,
    }
    return X, y, meta
