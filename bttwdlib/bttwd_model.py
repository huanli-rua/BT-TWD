import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

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

    def _build_bucket_estimator(self):
        """构建桶内局部模型（例如 KNN 或逻辑回归）。"""

        bcfg = self.cfg.get("BTTWD", {})
        est_name = bcfg.get("bucket_estimator", bcfg.get("posterior_estimator", "logreg"))
        if est_name == "knn":
            return KNeighborsClassifier(n_neighbors=bcfg.get("knn_k", 10))
        return LogisticRegression(max_iter=200, C=bcfg.get("logreg_C", 1.0))

    def _build_global_estimator(self):
        """构建全局后验估计器（例如 XGB）。"""

        bcfg = self.cfg.get("BTTWD", {})
        est_name = bcfg.get("global_estimator", None)
        if est_name is None:
            est_name = bcfg.get("bucket_estimator", bcfg.get("posterior_estimator", "logreg"))

        if est_name == "xgb":
            if not _XGB_AVAILABLE:
                raise RuntimeError("配置了 global_estimator='xgb' 但未安装 xgboost，请先安装该库。")
            xgb_cfg = bcfg.get("global_xgb", {})
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

        return self._build_bucket_estimator()

    def _find_model_with_backoff(self, bucket_id: str):
        """
        给定一个完整桶ID（如 'L1_age=mid|L2_education=high|L3_hours=normal_hours'），
        依次尝试：
        - 完整ID
        - 去掉最后一层
        - 再去掉一层
        直到找到已训练的桶模型；
        如果都找不到，则返回 (None, None)。
        """

        parts = bucket_id.split("|")
        for end in range(len(parts), 0, -1):
            candidate = "|".join(parts[:end])
            model = self.bucket_models.get(candidate)
            if model is not None:
                return model, candidate
        return None, None

    def fit(self, X: np.ndarray, y: np.ndarray, X_df_for_bucket: pd.DataFrame):
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        min_bucket_size = self.cfg.get("BTTWD", {}).get("min_bucket_size", 50)
        # 全局模型
        self.global_model = self._build_global_estimator()
        self.global_model.fit(X, y)
        log_info("【BTTWD】全局模型训练完成，用于兜底预测")

        grid_alpha = self.cfg.get("BTTWD", {}).get("threshold_search_grid", {}).get("alpha", [])
        grid_beta = self.cfg.get("BTTWD", {}).get("threshold_search_grid", {}).get("beta", [])
        threshold_obj = self.cfg.get("BTTWD", {}).get("threshold_objective", "F1")

        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            X_bucket = X[list(idxs)]
            y_bucket = y[list(idxs)]
            n_bucket = len(y_bucket)
            pos_rate = y_bucket.mean()

            # 1）样本数太少：构不成桶 → 用全局回退
            if n_bucket < min_bucket_size:
                log_info(
                    f"【BTTWD】桶 {bucket_id} 样本太少(n={n_bucket})，使用全局回退（min_bucket_size={min_bucket_size}）"
                )
                # 注意：这里不往 bucket_models 里放东西，让它预测时自动走全局
                continue

            # 2）单一类别桶：直接回退，不训练局部模型，避免 predict_proba 只有1列
            unique_classes = np.unique(y_bucket)
            if len(unique_classes) < 2:
                log_info(
                    f"【BTTWD】桶 {bucket_id} 仅包含单一类别 {int(unique_classes[0])} "
                    f"(n={n_bucket})，跳过局部模型训练，使用全局回退"
                )
                # 你也可以顺便记一下统计信息（可选）
                self.bucket_stats[bucket_id] = {
                    "bucket_id": bucket_id,
                    "n_samples": n_bucket,
                    "pos_rate": pos_rate,
                    "alpha": np.nan,
                    "beta": np.nan,
                    "train_score": np.nan,
                }
                continue

            # 3）正常桶：训练局部模型 + 做阈值搜索
            model = self._build_bucket_estimator()
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
                f"【BTTWD】桶 {bucket_id}：n={n_bucket}, pos_rate={pos_rate:.2f}, "
                f"alpha={best_alpha}, beta={best_beta}"
            )

        log_info(
            f"【BTTWD】共生成 {bucket_ids.nunique()} 个叶子桶，其中有效桶 {len(self.bucket_models)} 个（样本数 ≥ {min_bucket_size}）"
        )

    def predict_proba(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        proba = np.zeros(len(X))
        use_backoff = self.cfg.get("BTTWD", {}).get("use_global_backoff", True)
        verbose_backoff = self.cfg.get("EXP", {}).get("verbose_bucket_backoff", False)
        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            model, matched_bucket_id = self._find_model_with_backoff(bucket_id)
            if model is None and use_backoff:
                model = self.global_model
                matched_bucket_id = "GLOBAL"
            elif model is None:
                proba[list(idxs)] = self.global_alpha
                if verbose_backoff:
                    log_info(f"【BTTWD】桶 {bucket_id} 未找到回退模型，使用全局alpha={self.global_alpha}")
                continue
            if verbose_backoff and matched_bucket_id != bucket_id:
                log_info(f"【BTTWD】桶 {bucket_id} 回退到 {matched_bucket_id} 使用模型预测")
            proba[list(idxs)] = model.predict_proba(X[list(idxs)])[:, 1]
        return proba

    def predict(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X, X_df_for_bucket)
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        preds = np.zeros(len(proba))
        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            _, matched_bucket_id = self._find_model_with_backoff(bucket_id)
            if matched_bucket_id is not None:
                alpha, beta = self.bucket_thresholds.get(
                    matched_bucket_id, (self.global_alpha, self.global_beta)
                )
            else:
                alpha, beta = self.global_alpha, self.global_beta
            bucket_proba = proba[list(idxs)]
            bucket_pred = np.where(bucket_proba >= alpha, 1, np.where(bucket_proba <= beta, 0, -1))
            preds[list(idxs)] = bucket_pred
        return preds

    def get_bucket_stats(self) -> pd.DataFrame:
        if not self.bucket_stats:
            return pd.DataFrame()
        df = pd.DataFrame(self.bucket_stats.values())
        return df.sort_values(by="n_samples", ascending=False)
