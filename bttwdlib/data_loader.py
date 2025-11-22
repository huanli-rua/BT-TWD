import pandas as pd
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
