"""简易入口：使用 airlines 延误配置跑一次分层切分 + BTTWD 训练/评估。"""

from pathlib import Path
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from bttwdlib import (
    BucketTree,
    BTTWDModel,
    compute_binary_metrics,
    compute_s3_metrics,
    load_dataset,
    load_yaml_cfg,
    prepare_features_and_labels,
    show_cfg,
)
from bttwdlib.metrics import predict_binary_by_cost
from bttwdlib.utils_logging import log_info
from bttwdlib.utils_seed import set_global_seed


def _build_bucket_feature_df(df_raw, cfg) -> tuple[np.ndarray, np.ndarray, object, object, object]:
    X, y, meta = prepare_features_and_labels(df_raw, cfg)
    prep_cfg = cfg.get("PREPROCESS", {})
    bucket_cols: List[str] = (prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or [])
    bucket_df = df_raw[bucket_cols].reset_index(drop=True)
    return X, y, meta, bucket_df, bucket_cols


def main():
    cfg_path = Path("configs/airlines_delay.yaml")
    cfg = load_yaml_cfg(cfg_path)
    show_cfg(cfg)
    set_global_seed(cfg.get("SEED", {}).get("global_seed", 42))

    df_raw, target_col = load_dataset(cfg)
    log_info(
        f"【入口】数据集={cfg.get('DATA', {}).get('dataset_name')}，样本数={len(df_raw)}，标签列={target_col}"
    )

    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    log_info(f"【桶树层级】分裂顺序={[lvl.get('name') for lvl in bucket_levels]}")

    X, y, meta, bucket_df, bucket_cols = _build_bucket_feature_df(df_raw, cfg)

    split_cfg = cfg.get("DATA", {}).get("split", {})
    train_ratio = split_cfg.get("train_ratio", 0.7)
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.2)
    random_state = split_cfg.get("random_state", 42)

    X_train, X_temp, y_train, y_temp, bucket_train, bucket_temp = train_test_split(
        X,
        y,
        bucket_df,
        test_size=val_ratio + test_ratio,
        stratify=y,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test, bucket_val, bucket_test = train_test_split(
        X_temp,
        y_temp,
        bucket_temp,
        test_size=test_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.0,
        stratify=y_temp,
        random_state=random_state,
    )

    log_info(
        "【数据切分】训练/验证/测试样本数 = "
        f"{len(X_train)}/{len(X_val)}/{len(X_test)}，训练正类占比={y_train.mean():.2%}"
    )

    bucket_tree = BucketTree(bucket_levels, feature_names=bucket_cols)
    model = BTTWDModel(cfg, bucket_tree)
    model.fit(X_train, y_train, bucket_train)

    y_score = model.predict_proba(X_test, bucket_test)
    y_pred_s3 = model.predict(X_test, bucket_test)

    costs = cfg.get("THRESHOLDS", {}).get("costs", {})
    y_pred_binary = predict_binary_by_cost(y_score, costs) if costs else np.where(y_pred_s3 == 1, 1, 0)

    metrics_s3 = compute_s3_metrics(y_test, y_pred_s3, y_score, cfg.get("METRICS", {}), costs=costs)
    metrics_binary = compute_binary_metrics(y_test, y_pred_binary, y_score, cfg.get("METRICS", {}), costs=costs)

    log_info("【测试集指标-S3】" + ", ".join([f"{k}={v:.4f}" for k, v in metrics_s3.items()]))
    log_info("【测试集指标-二分类】" + ", ".join([f"{k}={v:.4f}" for k, v in metrics_binary.items()]))


if __name__ == "__main__":
    main()
