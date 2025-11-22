import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from .utils_logging import log_info


class BTTWDModel:
    def __init__(self, cfg: dict, bucket_tree):
        self.cfg = cfg
        self.bucket_tree = bucket_tree
        self.bucket_models = {}
        self.bucket_thresholds = {}
        self.bucket_stats = {}
        self.global_model = None
        self.global_alpha = cfg.get("BTTWD", {}).get("alpha_init", 0.6)
        self.global_beta = cfg.get("BTTWD", {}).get("beta_init", 0.3)

    def _build_estimator(self):
        bcfg = self.cfg.get("BTTWD", {})
        estimator = bcfg.get("posterior_estimator", "logreg")
        if estimator == "knn":
            return KNeighborsClassifier(n_neighbors=bcfg.get("knn_k", 10))
        return LogisticRegression(max_iter=200, C=bcfg.get("logreg_C", 1.0))

    def fit(self, X: np.ndarray, y: np.ndarray, X_df_for_bucket: pd.DataFrame):
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        min_bucket_size = self.cfg.get("BTTWD", {}).get("min_bucket_size", 50)
        # 全局模型
        self.global_model = self._build_estimator()
        self.global_model.fit(X, y)
        log_info("【BTTWD】全局模型训练完成，用于兜底预测")

        stats_rows = []
        grid_alpha = self.cfg.get("BTTWD", {}).get("threshold_search_grid", {}).get("alpha", [])
        grid_beta = self.cfg.get("BTTWD", {}).get("threshold_search_grid", {}).get("beta", [])
        threshold_obj = self.cfg.get("BTTWD", {}).get("threshold_objective", "F1")

        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            X_bucket = X[list(idxs)]
            y_bucket = y[list(idxs)]
            n_bucket = len(y_bucket)
            pos_rate = y_bucket.mean()
            if n_bucket < min_bucket_size:
                log_info(f"【BTTWD】桶 {bucket_id} 样本太少(n={n_bucket})，使用全局回退")
                continue
            model = self._build_estimator()
            model.fit(X_bucket, y_bucket)
            proba = model.predict_proba(X_bucket)[:, 1]

            best_alpha = self.global_alpha
            best_beta = self.global_beta
            best_score = -1
            if self.cfg.get("BTTWD", {}).get("optimize_thresholds", True):
                for a in grid_alpha:
                    for b in grid_beta:
                        if a < b:
                            continue
                        y_tmp = np.where(proba >= a, 1, np.where(proba <= b, 0, 0))
                        if threshold_obj == "BAC":
                            score = balanced_accuracy_score(y_bucket, y_tmp)
                        else:
                            score = f1_score(y_bucket, y_tmp)
                        if score > best_score:
                            best_score = score
                            best_alpha, best_beta = a, b

            self.bucket_models[bucket_id] = model
            self.bucket_thresholds[bucket_id] = (best_alpha, best_beta)
            self.bucket_stats[bucket_id] = {
                "bucket_id": bucket_id,
                "n_samples": n_bucket,
                "pos_rate": pos_rate,
                "alpha": best_alpha,
                "beta": best_beta,
                "train_score": best_score if best_score >= 0 else np.nan,
            }
            log_info(
                f"【BTTWD】桶 {bucket_id}：n={n_bucket}, pos_rate={pos_rate:.2f}, alpha={best_alpha}, beta={best_beta}"
            )
        log_info(f"【BTTWD】共生成 {bucket_ids.nunique()} 个叶子桶，其中有效桶 {len(self.bucket_models)} 个（样本数 ≥ {min_bucket_size}）")

    def predict_proba(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        proba = np.zeros(len(X))
        use_backoff = self.cfg.get("BTTWD", {}).get("use_global_backoff", True)
        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            model = self.bucket_models.get(bucket_id) if bucket_id in self.bucket_models else None
            if model is None and use_backoff:
                model = self.global_model
            elif model is None:
                proba[list(idxs)] = self.global_alpha
                continue
            proba[list(idxs)] = model.predict_proba(X[list(idxs)])[:, 1]
        return proba

    def predict(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X, X_df_for_bucket)
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        preds = np.zeros(len(proba))
        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            alpha, beta = self.bucket_thresholds.get(bucket_id, (self.global_alpha, self.global_beta))
            bucket_proba = proba[list(idxs)]
            bucket_pred = np.where(bucket_proba >= alpha, 1, np.where(bucket_proba <= beta, 0, -1))
            preds[list(idxs)] = bucket_pred
        return preds

    def get_bucket_stats(self) -> pd.DataFrame:
        if not self.bucket_stats:
            return pd.DataFrame()
        df = pd.DataFrame(self.bucket_stats.values())
        return df.sort_values(by="n_samples", ascending=False)
