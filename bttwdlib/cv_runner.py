import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False
from .bttwd_model import BTTWDModel
from .baselines import (
    get_decision_threshold,
    train_eval_knn,
    train_eval_logreg,
    train_eval_random_forest,
    train_eval_xgboost,
)
from .baseline_analyzer import run_baseline_bucket_evaluation
from .bucket_rules import BucketTree
from .metrics import (
    compute_binary_metrics,
    compute_s3_metrics,
    log_metrics,
    predict_binary_by_cost,
)
from .threshold_search import compute_regret
from .utils_logging import log_info


def _select_baselines(cfg: dict) -> set[str]:
    """根据配置确定需要运行的基线模型集合。"""

    model_cfg = cfg.get("MODEL", {})
    baseline_list = model_cfg.get("baselines") or []
    if isinstance(baseline_list, (list, tuple)) and baseline_list:
        return {str(x).lower() for x in baseline_list}

    base_cfg = cfg.get("BASELINES", {})
    selected = set()
    if base_cfg.get("use_logreg"):
        selected.add("logreg")
    if base_cfg.get("use_random_forest"):
        selected.add("random_forest")
    if base_cfg.get("use_knn"):
        selected.add("knn")
    if base_cfg.get("use_xgboost"):
        selected.add("xgb")
    return selected


def _build_baseline_estimator(model_key: str, cfg: dict):
    base_cfg = cfg.get("BASELINES", {})
    if model_key == "logreg":
        model_cfg = base_cfg.get("logreg", {})
        return LogisticRegression(max_iter=model_cfg.get("max_iter", 200), C=model_cfg.get("C", 1.0))
    if model_key == "random_forest":
        rf_cfg = base_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth"),
            random_state=rf_cfg.get("random_state", 42),
            n_jobs=cfg.get("EXP", {}).get("n_jobs", -1),
        )
    if model_key == "knn":
        knn_cfg = base_cfg.get("knn", {})
        return KNeighborsClassifier(n_neighbors=knn_cfg.get("n_neighbors", 10))
    if model_key in {"xgb", "xgboost"}:
        if not _XGB_AVAILABLE:
            raise RuntimeError("配置了 XGBoost 基线但未安装 xgboost，请先安装。")
        xgb_cfg = base_cfg.get("xgboost", {})
        return XGBClassifier(
            n_estimators=xgb_cfg.get("n_estimators", 300),
            max_depth=xgb_cfg.get("max_depth", 4),
            learning_rate=xgb_cfg.get("learning_rate", 0.1),
            subsample=xgb_cfg.get("subsample", 0.8),
            colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
            random_state=xgb_cfg.get("random_state", 42),
            n_jobs=xgb_cfg.get("n_jobs", -1),
            eval_metric="logloss",
            use_label_encoder=False,
        )
    raise ValueError(f"未知的基线模型类型 {model_key}")


def _eval_baseline_holdout(model_key: str, X_train, y_train, X_test, y_test, cfg, costs: dict | None = None) -> dict:
    clf = _build_baseline_estimator(model_key, cfg)
    threshold, mode, used_custom = get_decision_threshold(model_key if model_key != "xgb" else "xgboost", cfg)
    log_info(
        f"【基线-{model_key}】使用决策阈值={threshold:.3f}，模式={mode}，"
        f"自定义阈值={'是' if used_custom else '否'}，开始在测试集评估"
    )
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = clf.predict(X_test)
        y_score = np.zeros_like(y_pred, dtype=float)

    metrics_cfg = cfg.get("METRICS", {})
    metrics_dict = compute_binary_metrics(y_test, y_pred, y_score, metrics_cfg, costs=costs)
    metrics_dict.setdefault("BND_ratio", 0.0)
    metrics_dict.setdefault("POS_Coverage", float("nan"))
    metrics_dict["model"] = model_key
    return metrics_dict

def run_holdout_experiment(X, y, bucket_df, cfg, bucket_cols=None, bucket_tree: BucketTree | None = None, results_dir=None):
    """训练/评估单次切分的 BTTWD 模型并返回指标。"""

    repo_root = Path(__file__).resolve().parent.parent
    if results_dir is None:
        configured_results_dir = cfg.get("OUTPUT", {}).get("results_dir", "results")
        results_dir = Path(configured_results_dir)
        if not results_dir.is_absolute():
            results_dir = repo_root / results_dir

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

    # 重置分桶特征的索引，使其与对应的 X/y 数组位置对齐，避免后续按 index 访问概率时越界
    bucket_train = bucket_train.reset_index(drop=True)
    bucket_val = bucket_val.reset_index(drop=True)
    bucket_test = bucket_test.reset_index(drop=True)

    # 开发阶段的安全检查，确保长度一致
    assert len(X_train) == len(bucket_train) == len(y_train)
    assert len(X_val) == len(bucket_val) == len(y_val)
    assert len(X_test) == len(bucket_test) == len(y_test)

    log_info(
        "【数据切分】训练/验证/测试样本数 = "
        f"{len(X_train)}/{len(X_val)}/{len(X_test)}，训练正类占比={y_train.mean():.2%}"
    )

    bucket_cols = bucket_cols or bucket_df.columns.tolist()
    model = BTTWDModel.from_cfg(cfg, feature_names=bucket_cols)
    model.fit(X_train, y_train, bucket_train)

    y_score = model.predict_proba(X_test, bucket_test)
    y_pred_s3 = model.predict(X_test, bucket_test)

    costs = (cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})).get("costs", {})
    y_pred_binary = predict_binary_by_cost(y_score, costs) if costs else np.where(y_pred_s3 == 1, 1, 0)

    metrics_s3 = compute_s3_metrics(y_test, y_pred_s3, y_score, cfg.get("METRICS", {}), costs=costs)
    metrics_binary = compute_binary_metrics(y_test, y_pred_binary, y_score, cfg.get("METRICS", {}), costs=costs)

    log_info("【测试集指标-S3】" + ", ".join([f"{k}={v:.4f}" for k, v in metrics_s3.items()]))
    log_info("【测试集指标-二分类】" + ", ".join([f"{k}={v:.4f}" for k, v in metrics_binary.items()]))

    run_baseline_bucket_evaluation(
        X=X,
        y=y,
        bucket_df_for_split=bucket_df,
        bucket_tree=model.bucket_tree,
        cfg=cfg,
        results_dir=results_dir,
    )

    return {"metrics_s3": metrics_s3, "metrics_binary": metrics_binary}


def run_kfold_experiments(X, y, X_df_for_bucket, cfg, test_data=None, bucket_tree: BucketTree | None = None) -> dict:
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
    threshold_costs = (cfg.get("THRESHOLD") or cfg.get("THRESHOLDS", {})).get("costs", {})

    # 运行基线整体（使用 cross_val_predict）
    baseline_results = {}
    baseline_holdout_results = {}
    test_holdout_records: list[dict] = []
    baseline_set = _select_baselines(cfg)
    if "logreg" in baseline_set:
        baseline_results["LogReg"] = train_eval_logreg(X, y, cfg, skf, costs=threshold_costs)
    if "random_forest" in baseline_set:
        baseline_results["RandomForest"] = train_eval_random_forest(X, y, cfg, skf, costs=threshold_costs)
    if "knn" in baseline_set:
        baseline_results["KNN"] = train_eval_knn(X, y, cfg, skf, costs=threshold_costs)
    if "xgb" in baseline_set or "xgboost" in baseline_set:
        baseline_results["XGBoost"] = train_eval_xgboost(X, y, cfg, skf, costs=threshold_costs)

    fold_idx = 1
    model: BTTWDModel | None = None
    for train_idx, test_idx in skf.split(X, y):
        log_info(f"【K折实验】正在执行第 {fold_idx}/{n_splits} 折...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # 重新编号训练/测试数据的索引，确保与 X_train 对齐
        X_df_train = X_df_for_bucket.iloc[train_idx].reset_index(drop=True)
        X_df_test = X_df_for_bucket.iloc[test_idx].reset_index(drop=True)

        bttwd_model = BTTWDModel.from_cfg(cfg, feature_names=X_df_for_bucket.columns.tolist())
        bttwd_model.fit(X_train, y_train, X_df_train)

        model = bttwd_model

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
            test_bucket_ids = bttwd_model.bucket_tree.assign_buckets(X_df_test)
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

    if test_data is not None:
        X_test, y_test, bucket_df_test = test_data
        bucket_df_test = bucket_df_test.reset_index(drop=True)
        log_info(
            f"【Holdout】检测到外部测试集，训练集 n={len(X)}, 测试集 n={len(X_test)}，开始全量训练后评估"
        )
        bttwd_final = BTTWDModel.from_cfg(cfg, feature_names=X_df_for_bucket.columns.tolist())
        bttwd_final.fit(X, y, X_df_for_bucket.reset_index(drop=True))
        y_score_final = bttwd_final.predict_proba(X_test, bucket_df_test)
        y_pred_final = bttwd_final.predict(X_test, bucket_df_test)
        if threshold_costs:
            y_pred_binary_final = predict_binary_by_cost(y_score_final, threshold_costs)
        else:
            y_pred_binary_final = np.where(y_pred_final == 1, 1, 0)

        metrics_s3_test = compute_s3_metrics(
            y_test, y_pred_final, y_score_final, cfg.get("METRICS", {}), costs=threshold_costs
        )
        metrics_binary_test = compute_binary_metrics(
            y_test, y_pred_binary_final, y_score_final, cfg.get("METRICS", {}), costs=threshold_costs or None
        )
        metrics_s3_test.update({"model": "BTTWD", "fold": "test"})
        for k, v in metrics_binary_test.items():
            if k not in metrics_s3_test:
                metrics_s3_test[k] = v
        test_holdout_records.append(metrics_s3_test)
        per_fold_records.append(metrics_s3_test)
        log_metrics("【BTTWD-测试集】", metrics_s3_test)

        for base_key in baseline_set:
            res = _eval_baseline_holdout(base_key, X, y, X_test, y_test, cfg, costs=threshold_costs or None)
            res["fold"] = "test"
            baseline_holdout_results[base_key] = res
            per_fold_records.append(res)

    summary_df = pd.DataFrame(summary_rows)
    per_fold_output_df = pd.DataFrame(per_fold_records)

    overview_records = []
    if test_holdout_records or baseline_holdout_results:
        overview_records.extend(test_holdout_records)
        overview_records.extend(baseline_holdout_results.values())
    else:
        for row in summary_rows:
            base_row = {"model": row.get("model")}
            for k, v in row.items():
                if k.endswith("_mean"):
                    base_row[k[:-5]] = v
            overview_records.append(base_row)

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
    if overview_records:
        pd.DataFrame(overview_records).to_csv(os.path.join(results_dir, "metrics_overview.csv"), index=False)
    log_info("【K折实验】所有结果已写入 results 目录")

    bucket_tree_for_baseline = model.bucket_tree if model is not None else None

    run_baseline_bucket_evaluation(
        X=X,
        y=y,
        bucket_df_for_split=X_df_for_bucket,
        bucket_tree=bucket_tree_for_baseline,
        cfg=cfg,
        results_dir=results_dir,
    )

    return {"baselines": baseline_results, "bttwd": {"per_fold": per_fold_records, "summary": summary_rows}}
