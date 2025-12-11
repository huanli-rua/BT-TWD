import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from .utils_logging import log_info


def _apply_missing_handling(df: pd.DataFrame, prep_cfg: dict) -> pd.DataFrame:
    """Apply lightweight missing-value preprocessing that must happen before fitting/transforming."""

    df = df.copy()
    handle_missing = prep_cfg.get("handle_missing")
    if handle_missing == "question_mark":
        df.replace("?", np.nan, inplace=True)

    if handle_missing == "simple" and prep_cfg.get("fillna_strategy", "most_frequent") == "drop":
        df = df.dropna().reset_index(drop=True)

    return df


def _build_simple_imputer(strategy: str, feature_type: str):
    if strategy == "most_frequent":
        return SimpleImputer(strategy="most_frequent")
    if strategy == "median" and feature_type == "numeric":
        return SimpleImputer(strategy="median")
    if strategy == "mean" and feature_type == "numeric":
        return SimpleImputer(strategy="mean")
    if strategy == "zero":
        return SimpleImputer(strategy="constant", fill_value=0)
    return None


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
    target_transform = data_cfg.get("target_transform") or {}
    target_col_for_model = target_transform.get("new_col", target_col)
    positive_label = data_cfg.get("positive_label", ">50K")
    negative_label = data_cfg.get("negative_label")

    df = _apply_missing_handling(df, prep_cfg)
    handle_missing = prep_cfg.get("handle_missing")
    strategy = None
    if handle_missing == "simple":
        strategy = prep_cfg.get("fillna_strategy", "most_frequent")
        log_info(f"【预处理】缺失值填充策略={strategy}")

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
        continuous_cols, categorical_cols = _infer_columns(df, target_col_for_model)
    log_info(f"【预处理】连续特征={len(continuous_cols)}个，类别特征={len(categorical_cols)}个")

    target_series = df[target_col_for_model]
    # 若目标列已经是 0/1 数值标签，直接复用，避免二次转换导致全为 0
    if set(pd.unique(target_series.dropna())) <= {0, 1}:
        y = target_series.astype(int).values
    else:
        y = (target_series == positive_label).astype(int).values
        if negative_label is not None:
            y = np.where(target_series == positive_label, 1, 0)
    drop_cols = set(prep_cfg.get("drop_cols", []))
    drop_cols.add(target_col_for_model)
    source_target_col = data_cfg.get("target_col")
    if source_target_col and source_target_col != target_col_for_model:
        drop_cols.add(source_target_col)
    X_raw = df.drop(columns=list(drop_cols), errors="ignore")

    transformers = []
    cat_imputer = None
    num_imputer = None
    if handle_missing == "simple" and strategy != "drop":
        cat_imputer = _build_simple_imputer(strategy, "categorical") if categorical_cols else None
        num_imputer = _build_simple_imputer(strategy, "numeric") if continuous_cols else None
    if categorical_cols:
        encoder = OneHotEncoder(drop="first" if prep_cfg.get("drop_first") else None, handle_unknown="ignore")
        cat_steps = []
        if cat_imputer is not None:
            cat_steps.append(("imputer", cat_imputer))
        cat_steps.append(("encoder", encoder))
        cat_pipeline = Pipeline(steps=cat_steps)
        transformers.append(("cat", cat_pipeline, categorical_cols))
    if continuous_cols:
        scaler_type = prep_cfg.get("scaler_type", "standard")
        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        num_steps = []
        if num_imputer is not None:
            num_steps.append(("imputer", num_imputer))
        num_steps.append(("scaler", scaler))
        num_pipeline = Pipeline(steps=num_steps)
        transformers.append(("num", num_pipeline, continuous_cols))

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
