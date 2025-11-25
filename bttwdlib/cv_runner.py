import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from .bttwd_model import BTTWDModel
from .bucket_rules import BucketTree
from .baselines import (
    train_eval_knn,
    train_eval_logreg,
    train_eval_random_forest,
    train_eval_xgboost,
)
from .metrics import (
    compute_binary_metrics,
    compute_s3_metrics,
    log_metrics,
    predict_binary_by_cost,
)
from .threshold_search import compute_regret
from .utils_logging import log_info


def run_holdout_experiment(X, y, bucket_df, cfg, bucket_cols=None):
    """训练/评估单次切分的 BTTWD 模型并返回指标。"""

    split_cfg = cfg.get("DATA", {}).get("split", {})
    val_ratio = split_cfg.get("val_ratio", 0.1)
    test_ratio = split_cfg.get("test_ratio", 0.2)
    random_state = split_cfg.get("random_state", 42)
    from sklearn.model_selection import train_test_split

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

    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    bucket_cols = bucket_cols or bucket_df.columns.tolist()
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

    return {"metrics_s3": metrics_s3, "metrics_binary": metrics_binary}


def run_kfold_experiments(X, y, X_df_for_bucket, cfg) -> dict:
    repo_root = Path(__file__).resolve().parent.parent
    configured_results_dir = cfg.get("OUTPUT", {}).get("results_dir", "results")
    results_dir = Path(configured_results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    os.makedirs(results_dir, exist_ok=True)

    n_splits = cfg.get("DATA", {}).get("n_splits", 5)
    shuffle = cfg.get("DATA", {}).get("shuffle", True)
    random_state = cfg.get("DATA", {}).get("random_state", 42)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    per_fold_records = []
    bucket_metrics_records = []
    threshold_log_records = []
    threshold_costs = cfg.get("THRESHOLDS", {}).get("costs", {})

    # 运行基线整体（使用 cross_val_predict）
    baseline_results = {}
    if cfg.get("BASELINES", {}).get("use_logreg", False):
        baseline_results["LogReg"] = train_eval_logreg(X, y, cfg, skf, costs=threshold_costs)
    if cfg.get("BASELINES", {}).get("use_random_forest", False):
        baseline_results["RandomForest"] = train_eval_random_forest(X, y, cfg, skf, costs=threshold_costs)
    if cfg.get("BASELINES", {}).get("use_knn", False):
        baseline_results["KNN"] = train_eval_knn(X, y, cfg, skf, costs=threshold_costs)
    if cfg.get("BASELINES", {}).get("use_xgboost", False):
        baseline_results["XGBoost"] = train_eval_xgboost(X, y, cfg, skf, costs=threshold_costs)

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
        if threshold_costs:
            y_pred_binary = predict_binary_by_cost(y_score, threshold_costs)
        else:
            y_pred_binary = np.where(y_pred_s3 == 1, 1, 0)

        metrics_binary = compute_binary_metrics(
            y_test, y_pred_binary, y_score, cfg.get("METRICS", {}), costs=threshold_costs or None
        )
        metrics_s3 = compute_s3_metrics(y_test, y_pred_s3, y_score, cfg.get("METRICS", {}), costs=threshold_costs)
        log_metrics("【BTTWD】三支指标(含后悔)：", metrics_s3)

        fold_record = {"fold": fold_idx, "model": "BTTWD", **metrics_s3}
        for k, v in metrics_binary.items():
            if k not in fold_record:
                fold_record[k] = v
        per_fold_records.append(fold_record)
        bucket_df = bttwd_model.get_bucket_stats()
        if not bucket_df.empty:
            test_bucket_ids = bucket_tree.assign_buckets(X_df_test)
            test_bucket_records = []
            for bucket_id, idxs in test_bucket_ids.groupby(test_bucket_ids).groups.items():
                idx_list = list(idxs)
                y_true_bucket = y_test[idx_list]
                y_pred_bucket = y_pred_s3[idx_list]
                test_bucket_records.append(
                    {
                        "bucket_id": bucket_id,
                        "n_test": len(idx_list),
                        "pos_rate_test": float(np.mean(y_true_bucket)) if len(idx_list) else np.nan,
                        "BND_ratio_test": float(np.mean(np.isin(y_pred_bucket, [-1, "BND"]))),
                        "POS_Coverage_test": float(np.mean(np.array(y_pred_bucket) == 1)),
                        "regret_test": compute_regret(y_true_bucket, y_pred_bucket, threshold_costs),
                    }
                )

            test_bucket_df = pd.DataFrame(test_bucket_records)
            if not test_bucket_df.empty:
                bucket_df = bucket_df.merge(test_bucket_df, on="bucket_id", how="left")
            else:
                bucket_df["n_test"] = np.nan
                bucket_df["pos_rate_test"] = np.nan
                bucket_df["BND_ratio_test"] = np.nan
                bucket_df["POS_Coverage_test"] = np.nan
                bucket_df["regret_test"] = np.nan
            bucket_df["fold"] = fold_idx
            bucket_metrics_records.append(bucket_df)

        th_logs = bttwd_model.get_threshold_logs()
        if not th_logs.empty:
            th_logs["fold"] = fold_idx
            threshold_log_records.append(th_logs)

        fold_idx += 1

    # 汇总 BTTWD 平均指标
    bttwd_df = pd.DataFrame(per_fold_records)
    summary_rows = []
    if not bttwd_df.empty:
        metric_cols = [c for c in bttwd_df.columns if c not in ["fold", "model"]]
        mean_series = bttwd_df[metric_cols].mean()
        std_series = bttwd_df[metric_cols].std()
        bttwd_summary = {"model": "BTTWD"}
        for col in metric_cols:
            bttwd_summary[f"{col}_mean"] = mean_series[col]
            bttwd_summary[f"{col}_std"] = std_series[col]
        summary_rows.append(bttwd_summary)

    for model_name, res in baseline_results.items():
        if res["summary"]:
            row = {"model": model_name}
            row.update(res["summary"])
            summary_rows.append(row)

        if res.get("per_fold"):
            for rec in res["per_fold"]:
                per_fold_records.append({"model": model_name, **rec})

    summary_df = pd.DataFrame(summary_rows)
    per_fold_output_df = pd.DataFrame(per_fold_records)

    # 写文件
    if cfg.get("OUTPUT", {}).get("save_per_fold_metrics", True):
        per_fold_output_df.to_csv(os.path.join(results_dir, "metrics_kfold_per_fold.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "metrics_kfold_summary.csv"), index=False)
    if bucket_metrics_records and cfg.get("OUTPUT", {}).get("save_bucket_metrics", True):
        all_bucket_df = pd.concat(bucket_metrics_records, ignore_index=True)
        if "pos_rate_all" in all_bucket_df.columns and "pos_rate" not in all_bucket_df.columns:
            all_bucket_df["pos_rate"] = all_bucket_df["pos_rate_all"]
        all_bucket_df.to_csv(os.path.join(results_dir, "bucket_metrics.csv"), index=False)
    if threshold_log_records and cfg.get("OUTPUT", {}).get("save_threshold_logs", True):
        th_filename = cfg.get("OUTPUT", {}).get("threshold_log_filename", "bucket_thresholds_per_fold.csv")
        pd.concat(threshold_log_records, ignore_index=True).to_csv(
            os.path.join(results_dir, th_filename), index=False
        )
    log_info("【K折实验】所有结果已写入 results 目录")

    return {"baselines": baseline_results, "bttwd": {"per_fold": per_fold_records, "summary": summary_rows}}
