from pathlib import Path

import pandas as pd
from scipy.io import arff
from .utils_logging import log_info


def load_adult_raw(cfg: dict) -> pd.DataFrame:
    """
    从 cfg['DATA']['raw_path'] 读取 CSV，将 "?" 视为缺失值。
    返回 DataFrame，包含列名。
    """
    data_cfg = cfg.get("DATA", {})
    path = data_cfg.get("raw_path")
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        data_cfg.get("target_col", "income"),
    ]
    df = pd.read_csv(
        path,
        header=None,
        names=col_names,
        na_values=["?"],
        skipinitialspace=True,
    )
    target_col = data_cfg.get("target_col", "income")
    pos_label = data_cfg.get("positive_label", ">50K")
    total = len(df)
    n_features = df.shape[1] - 1
    pos_rate = (df[target_col] == pos_label).mean()
    log_info(
        f"【数据加载完毕】样本数={total}，特征数={n_features}，正类比例={pos_rate:.2f}"
    )
    return df


def _load_csv_like(path: str, data_cfg: dict) -> pd.DataFrame:
    sep = data_cfg.get("sep", ",")
    encoding = data_cfg.get("encoding", "utf-8")
    header = data_cfg.get("header", "infer")
    names = data_cfg.get("col_names")
    skiprows = data_cfg.get("skiprows")
    df = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        header=header,
        names=names,
        skiprows=skiprows,
    )
    log_info(f"【数据加载】文本表格 {path} 已读取，样本数={len(df)}，列数={df.shape[1]}")
    return df


def _load_arff(path: str) -> pd.DataFrame:
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    # 将 bytes 类型的类别值解码为字符串
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, (bytes, bytearray)) else x)
    log_info(f"【数据加载】ARFF 文件 {path} 已读取，含 {df.shape[0]} 条记录，{df.shape[1]} 列")
    return df


def _apply_target_transform(df: pd.DataFrame, data_cfg: dict) -> tuple[pd.DataFrame, str]:
    target_col = data_cfg.get("target_col")
    transform_cfg = data_cfg.get("target_transform") or {}
    if not transform_cfg:
        return df, target_col

    transform_type = transform_cfg.get("type")
    if transform_type == "threshold_binary":
        threshold = transform_cfg.get("threshold", 0.0)
        greater_is_positive = transform_cfg.get("greater_is_positive", True)
        new_col = transform_cfg.get("new_col", f"{target_col}_bin")
        cmp = df[target_col] > threshold if greater_is_positive else df[target_col] < threshold
        df[new_col] = cmp.astype(int)
        log_info(
            f"【目标变换】已按阈值 {threshold} 生成二分类标签列 {new_col}，正类取 {'>' if greater_is_positive else '<'} {threshold}"
        )
        return df, new_col

    log_info(f"【目标变换】未识别的 target_transform.type={transform_type}，保持原目标列 {target_col}")
    return df, target_col


def load_dataset(cfg: dict) -> tuple[pd.DataFrame, str]:
    """根据配置加载数据集，支持 adult CSV、ARFF 以及多种表格格式。"""

    data_cfg = cfg.get("DATA", {})
    raw_path = data_cfg.get("raw_path") or data_cfg.get("path") or data_cfg.get("data_path")
    file_type_cfg = data_cfg.get("file_type")
    file_type = str(file_type_cfg).lower() if file_type_cfg is not None else ""
    if raw_path:
        raw_path_path = Path(raw_path)
        if not raw_path_path.is_absolute() and not raw_path_path.exists():
            repo_root = Path(__file__).resolve().parent.parent
            alt_path = repo_root / raw_path_path
            if alt_path.exists():
                raw_path_path = alt_path
        raw_path = str(raw_path_path)

    if not file_type and raw_path:
        file_type = Path(raw_path).suffix.lower().lstrip(".") or "csv"
    dataset_name = data_cfg.get("dataset_name", "dataset")

    if raw_path is None:
        raise FileNotFoundError("配置中缺少 raw_path/path 字段，无法读取数据")

    if file_type == "arff":
        df = _load_arff(raw_path)
    elif file_type in {"csv", "txt"}:
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower == "adult":
            df = load_adult_raw(cfg)
        elif dataset_name_lower == "bank_full":
            tmp_cfg = dict(data_cfg)
            tmp_cfg.setdefault("sep", ";")
            df = _load_csv_like(raw_path, tmp_cfg)
            target_col = data_cfg.get("target_col", "y")
            if target_col not in df.columns:
                raise KeyError(f"银行数据集中未找到标签列 {target_col}")
            y_raw = df[target_col].astype(str).str.strip().str.lower()
            df[target_col] = y_raw.map({"yes": 1, "no": 0})
            if df[target_col].isna().any():
                raise ValueError("银行数据集的标签列存在无法识别的取值（非 yes/no）")
            df[target_col] = df[target_col].astype(int)
            data_cfg["positive_label"] = 1
            data_cfg.setdefault("negative_label", 0)
            data_cfg.setdefault(
                "numeric_cols",
                ["age", "balance", "day", "duration", "campaign", "pdays", "previous"],
            )
            data_cfg.setdefault(
                "categorical_cols",
                [
                    "job",
                    "marital",
                    "education",
                    "default",
                    "housing",
                    "loan",
                    "contact",
                    "month",
                    "poutcome",
                ],
            )
            log_info(
                "【数据加载】银行营销数据集已读取，标签已映射为0/1，"
                f"样本数={len(df)}，正类比例={df[target_col].mean():.2%}"
            )
        else:
            df = _load_csv_like(raw_path, data_cfg)
    elif file_type == "tsv":
        tmp = dict(data_cfg)
        tmp.setdefault("sep", "\t")
        df = _load_csv_like(raw_path, tmp)
    elif file_type in {"dat", "data"}:
        tmp = dict(data_cfg)
        tmp.setdefault("sep", None)
        df = _load_csv_like(raw_path, tmp)
    elif file_type == "parquet":
        df = pd.read_parquet(raw_path)
    elif file_type in {"feather"}:
        df = pd.read_feather(raw_path)
    elif file_type in {"excel", "xlsx", "xls"}:
        sheet = data_cfg.get("sheet_name", 0)
        df = pd.read_excel(raw_path, sheet_name=sheet)
    elif file_type in {"json", "jsonl"}:
        df = pd.read_json(raw_path, lines=True)
    else:
        raise ValueError(f"未知的 file_type={file_type}")

    df, target_col = _apply_target_transform(df, data_cfg)

    positive_label = data_cfg.get("positive_label")
    if positive_label is not None and target_col in df.columns:
        pos_rate = (df[target_col] == positive_label).mean()
        log_info(
            f"【数据集信息】名称={dataset_name}，样本数={len(df)}，目标列={target_col}，正类比例={pos_rate:.2%}"
        )
    else:
        log_info(f"【数据集信息】名称={dataset_name}，样本数={len(df)}，目标列={target_col}")

    return df, target_col
