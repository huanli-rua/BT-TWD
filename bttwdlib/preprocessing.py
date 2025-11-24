import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .utils_logging import log_info


def _infer_columns(df: pd.DataFrame, target_col: str, extra_excludes: list | None = None):
    excludes = {target_col}
    if extra_excludes:
        excludes.update(extra_excludes)
    df_no_target = df.drop(columns=list(excludes), errors="ignore")
    continuous_cols = df_no_target.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df_no_target.select_dtypes(exclude=["number"]).columns.tolist()
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
    target_transform = data_cfg.get("target_transform") or {}
    target_col_for_model = target_transform.get("new_col", target_col)
    positive_label = data_cfg.get("positive_label", ">50K")
    negative_label = data_cfg.get("negative_label")

    df = df.copy()
    # 缺失值统一处理
    if prep_cfg.get("handle_missing") == "question_mark":
        df.replace("?", np.nan, inplace=True)

    # 推断列
    continuous_cols = (
        prep_cfg.get("continuous_cols")
        or prep_cfg.get("numeric")
        or prep_cfg.get("numeric_cols")
        or []
    )
    categorical_cols = (
        prep_cfg.get("categorical_cols")
        or prep_cfg.get("categorical")
        or []
    )
    if not continuous_cols and not categorical_cols:
        continuous_cols, categorical_cols = _infer_columns(
            df, target_col_for_model, extra_excludes=[target_col]
        )
    log_info(f"【预处理】连续特征={len(continuous_cols)}个，类别特征={len(categorical_cols)}个")

    y = (df[target_col_for_model] == positive_label).astype(int).values
    if negative_label is not None:
        y = np.where(df[target_col_for_model] == positive_label, 1, 0)
    drop_cols = set(prep_cfg.get("drop_cols", []))
    drop_cols.add(target_col_for_model)
    source_target_col = data_cfg.get("target_col")
    if source_target_col and source_target_col != target_col_for_model:
        drop_cols.add(source_target_col)
    X_raw = df.drop(columns=list(drop_cols), errors="ignore")

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

    # 确保输出的特征矩阵 X 为可写的 numpy 数组（避免后续模型报 buffer read-only 错误）
    if sparse.issparse(X):
        X = X.toarray()
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
