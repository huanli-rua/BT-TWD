import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .bttwd_model import BTTWDModel
from .bucket_rules import BucketTree
from .baselines import train_eval_logreg, train_eval_random_forest
from .metrics import compute_binary_metrics, compute_s3_metrics, log_metrics
from .utils_logging import log_info


def run_kfold_experiments(X, y, X_df_for_bucket, cfg) -> dict:
    results_dir = cfg.get("OUTPUT", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    n_splits = cfg.get("DATA", {}).get("n_splits", 5)
    shuffle = cfg.get("DATA", {}).get("shuffle", True)
    random_state = cfg.get("DATA", {}).get("random_state", 42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    per_fold_records = []
    bucket_metrics_records = []

    # 运行基线整体（使用 cross_val_predict）
    baseline_results = {}
    if cfg.get("BASELINES", {}).get("use_logreg", False):
        baseline_results["LogReg"] = train_eval_logreg(X, y, cfg, skf)
    if cfg.get("BASELINES", {}).get("use_random_forest", False):
        baseline_results["RandomForest"] = train_eval_random_forest(X, y, cfg, skf)

    fold_idx = 1
    for train_idx, test_idx in skf.split(X, y):
        log_info(f"【K折实验】正在执行第 {fold_idx}/{n_splits} 折...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 重新编号训练/测试数据的索引，确保与 X_train 对齐
        X_df_train = X_df_for_bucket.iloc[train_idx].reset_index(drop=True)
        X_df_test = X_df_for_bucket.iloc[test_idx].reset_index(drop=True)

        bucket_tree = BucketTree(cfg.get("BTTWD", {}).get("bucket_levels", []), feature_names=X_df_for_bucket.columns.tolist())
        bttwd_model = BTTWDModel(cfg, bucket_tree)
        bttwd_model.fit(X_train, y_train, X_df_train)

        y_score = bttwd_model.predict_proba(X_test, X_df_test)
        y_pred_s3 = bttwd_model.predict(X_test, X_df_test)
        y_pred_binary = np.where(y_pred_s3 == 1, 1, 0)

        metrics_binary = compute_binary_metrics(y_test, y_pred_binary, y_score, cfg.get("METRICS", {}))
        metrics_s3 = compute_s3_metrics(y_test, y_pred_s3, y_score, cfg.get("METRICS", {}))
        log_metrics("【BTTWD】本折指标：", metrics_binary)

        per_fold_records.append(
            {
                "fold": fold_idx,
                "model": "BTTWD",
                **metrics_binary,
            }
        )
        for model_name, res in baseline_results.items():
            if res["per_fold"] is None:
                continue
        bucket_df = bttwd_model.get_bucket_stats()
        if not bucket_df.empty:
            bucket_df["fold"] = fold_idx
            bucket_metrics_records.append(bucket_df)

        fold_idx += 1

    # 汇总 BTTWD 平均指标
    bttwd_df = pd.DataFrame(per_fold_records)
    summary_rows = []
    if not bttwd_df.empty:
        bttwd_summary = bttwd_df.drop(columns=["fold", "model"]).mean().to_dict()
        bttwd_summary["model"] = "BTTWD"
        summary_rows.append(bttwd_summary)

    for model_name, res in baseline_results.items():
        if res["summary"]:
            row = {"model": model_name}
            row.update(res["summary"])
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # 写文件
    if cfg.get("OUTPUT", {}).get("save_per_fold_metrics", True):
        bttwd_df.to_csv(os.path.join(results_dir, "metrics_kfold_per_fold.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "metrics_kfold_summary.csv"), index=False)
    if bucket_metrics_records and cfg.get("OUTPUT", {}).get("save_bucket_metrics", True):
        all_bucket_df = pd.concat(bucket_metrics_records, ignore_index=True)
        all_bucket_df.to_csv(os.path.join(results_dir, "bucket_metrics.csv"), index=False)
    log_info("【K折实验】所有结果已写入 results 目录")

    return {"baselines": baseline_results, "bttwd": {"per_fold": per_fold_records, "summary": summary_rows}}
