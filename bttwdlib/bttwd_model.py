import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .bucket_rules import get_parent_bucket_id
from .utils_logging import log_info


class BTTWDModel:
    def __init__(self, cfg: dict, bucket_tree):
        self.cfg = cfg
        self.bucket_tree = bucket_tree

        bcfg = cfg.get("BTTWD", {})
        data_cfg = cfg.get("DATA", {})

        self.min_bucket_size = bcfg.get("min_bucket_size", 50)
        self.parent_share_rate = bcfg.get("parent_share_rate", 0.0)
        self.min_parent_share = bcfg.get("min_parent_share", 0)
        self.val_ratio = bcfg.get("val_ratio", 0.2)
        self.min_val_samples_per_bucket = bcfg.get("min_val_samples_per_bucket", 10)
        self.use_global_backoff = bcfg.get("use_global_backoff", True)
        self.optimize_thresholds = bcfg.get("optimize_thresholds", True)
        self.threshold_obj = bcfg.get("threshold_obj", bcfg.get("threshold_objective", "F1"))
        threshold_grid_cfg = bcfg.get("threshold_grid", bcfg.get("threshold_search_grid", {}))
        self.threshold_grid_alpha = threshold_grid_cfg.get("alpha", [])
        self.threshold_grid_beta = threshold_grid_cfg.get("beta", [])
        self.global_alpha = bcfg.get("alpha_init", 0.6)
        self.global_beta = bcfg.get("beta_init", 0.3)
        self.random_state = data_cfg.get("random_state", 42)

        self.bucket_models = {}
        self.bucket_thresholds = {}
        self.bucket_stats = {}
        self.global_model = None
        self.global_pos_rate = 0.5
        self.rng = np.random.default_rng(self.random_state)

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
        """逐级回退查找桶模型。"""

        parts = bucket_id.split("|")
        for end in range(len(parts), 0, -1):
            candidate = "|".join(parts[:end])
            model = self.bucket_models.get(candidate)
            if model is not None:
                return model, candidate
        return None, None

    def _search_thresholds(self, proba: np.ndarray, y_true: np.ndarray):
        grid_alpha = self.threshold_grid_alpha or [self.global_alpha]
        grid_beta = self.threshold_grid_beta or [self.global_beta]

        best_alpha = self.global_alpha
        best_beta = self.global_beta
        best_score = -np.inf

        for alpha in grid_alpha:
            for beta in grid_beta:
                if alpha < beta:
                    continue
                y_tmp = np.where(proba >= alpha, 1, np.where(proba <= beta, 0, 0))
                if self.threshold_obj.upper() == "BAC":
                    score = balanced_accuracy_score(y_true, y_tmp)
                else:
                    score = f1_score(y_true, y_tmp)
                if score > best_score:
                    best_score = score
                    best_alpha, best_beta = alpha, beta
        return best_alpha, best_beta, best_score

    def fit(self, X: np.ndarray, y: np.ndarray, X_df_for_bucket: pd.DataFrame):
        # Step 0: 划分 inner 训练/验证集
        if self.val_ratio > 0:
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=self.val_ratio, random_state=self.random_state
            )
            inner_train_idx, inner_val_idx = next(sss.split(X, y))
        else:
            inner_train_idx = np.arange(len(y))
            inner_val_idx = np.array([], dtype=int)

        inner_train_idx = np.asarray(inner_train_idx)
        inner_val_idx = np.asarray(inner_val_idx)

        self.global_pos_rate = float(np.mean(y))

        # Step 1: 构建叶子与父桶的样本映射
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        df_buckets = pd.DataFrame({"idx": np.arange(len(y)), "bucket_id": bucket_ids, "y": y})

        leaf_index_map = {}
        parent_index_map = defaultdict(list)

        for bucket_id, g in df_buckets.groupby("bucket_id"):
            idx_all = g["idx"].to_numpy()
            y_bucket = g["y"].to_numpy()
            n_bucket = len(idx_all)

            parent_id = get_parent_bucket_id(bucket_id)

            if n_bucket < self.min_bucket_size:
                if parent_id is not None:
                    parent_index_map[parent_id].extend(idx_all.tolist())
                    log_info(
                        f"【BTTWD】桶 {bucket_id} 样本太少(n={n_bucket})，全部并入父桶 {parent_id}"
                    )
                else:
                    log_info(
                        f"【BTTWD】顶层桶 {bucket_id} 样本太少(n={n_bucket})，仅使用全局模型兜底"
                    )
                continue

            leaf_index_map[bucket_id] = idx_all

            if parent_id is not None:
                n_share = max(int(n_bucket * self.parent_share_rate), self.min_parent_share)
                n_share = min(n_share, n_bucket)
                share_idx = self.rng.choice(idx_all, size=n_share, replace=False)
                parent_index_map[parent_id].extend(share_idx.tolist())
                log_info(f"【BTTWD】桶 {bucket_id} 向父桶 {parent_id} 贡献 {len(share_idx)} 个典型样本")

        # Step 2: 训练全局模型 + 阈值
        X_train_inner = X[inner_train_idx]
        y_train_inner = y[inner_train_idx]

        self.global_model = self._build_global_estimator()
        self.global_model.fit(X_train_inner, y_train_inner)
        log_info("【BTTWD】全局模型训练完成，用于兜底预测")

        if self.optimize_thresholds and len(inner_val_idx) > 0:
            X_val_inner = X[inner_val_idx]
            y_val_inner = y[inner_val_idx]
            proba_val_inner = self.global_model.predict_proba(X_val_inner)[:, 1]
            self.global_alpha, self.global_beta, _ = self._search_thresholds(proba_val_inner, y_val_inner)

        # Step 3: 训练父桶模型
        self.bucket_models = {}
        self.bucket_thresholds = {}
        self.bucket_stats = {}

        for parent_id, idx_list in parent_index_map.items():
            idx_all = np.array(sorted(set(idx_list)))
            y_all = y[idx_all]

            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)

            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            y_train_bucket = y[train_idx_bucket]
            y_val_bucket = y[val_idx_bucket]

            if len(y_train_bucket) < self.min_bucket_size or np.unique(y_train_bucket).size < 2:
                log_info(f"【BTTWD】父桶 {parent_id} 训练样本不足或单类，跳过局部模型训练")
                continue

            model = self._build_bucket_estimator()
            model.fit(X[train_idx_bucket], y_train_bucket)

            use_full_bucket_for_threshold = False
            if (
                len(val_idx_bucket) >= self.min_val_samples_per_bucket
                and np.unique(y_val_bucket).size >= 2
            ):
                proba_val = model.predict_proba(X[val_idx_bucket])[:, 1]
                alpha, beta, score = self._search_thresholds(proba_val, y_val_bucket)
            else:
                proba_all = model.predict_proba(X[idx_all])[:, 1]
                alpha, beta, score = self._search_thresholds(proba_all, y_all)
                use_full_bucket_for_threshold = True

            self.bucket_models[parent_id] = model
            self.bucket_thresholds[parent_id] = (alpha, beta)
            self.bucket_stats[parent_id] = {
                "bucket_id": parent_id,
                "level": "parent",
                "n_samples_all": int(len(idx_all)),
                "n_samples_train": int(len(train_idx_bucket)),
                "n_samples_val": int(len(val_idx_bucket)),
                "pos_rate": float(y_all.mean()),
                "alpha": float(alpha),
                "beta": float(beta),
                "threshold_score": float(score),
                "use_full_bucket_for_threshold": bool(use_full_bucket_for_threshold),
            }

        # Step 4: 训练叶子桶模型
        for bucket_id, idx_all in leaf_index_map.items():
            idx_all = np.asarray(idx_all)
            y_all = y[idx_all]

            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)

            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            y_train_bucket = y[train_idx_bucket]
            y_val_bucket = y[val_idx_bucket]

            if len(y_train_bucket) < self.min_bucket_size or np.unique(y_train_bucket).size < 2:
                log_info(f"【BTTWD】叶子桶 {bucket_id} 训练样本不足或单类，跳过局部模型训练")
                continue

            model = self._build_bucket_estimator()
            model.fit(X[train_idx_bucket], y_train_bucket)

            use_full_bucket_for_threshold = False
            if (
                len(val_idx_bucket) >= self.min_val_samples_per_bucket
                and np.unique(y_val_bucket).size >= 2
            ):
                proba_val = model.predict_proba(X[val_idx_bucket])[:, 1]
                alpha, beta, score = self._search_thresholds(proba_val, y_val_bucket)
            else:
                proba_all = model.predict_proba(X[idx_all])[:, 1]
                alpha, beta, score = self._search_thresholds(proba_all, y_all)
                use_full_bucket_for_threshold = True

            self.bucket_models[bucket_id] = model
            self.bucket_thresholds[bucket_id] = (alpha, beta)
            self.bucket_stats[bucket_id] = {
                "bucket_id": bucket_id,
                "level": "leaf",
                "n_samples_all": int(len(idx_all)),
                "n_samples_train": int(len(train_idx_bucket)),
                "n_samples_val": int(len(val_idx_bucket)),
                "pos_rate": float(y_all.mean()),
                "alpha": float(alpha),
                "beta": float(beta),
                "threshold_score": float(score),
                "use_full_bucket_for_threshold": bool(use_full_bucket_for_threshold),
            }

        log_info(
            f"【BTTWD】共生成 {bucket_ids.nunique()} 个叶子桶，其中有效桶 {len(self.bucket_models)} 个（样本数 ≥ {self.min_bucket_size}）"
        )

    def predict_proba(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        proba = np.zeros(len(X))

        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            model, matched_bucket_id = self._find_model_with_backoff(bucket_id)

            if model is None:
                if self.use_global_backoff and self.global_model is not None:
                    proba[list(idxs)] = self.global_model.predict_proba(X[list(idxs)])[:, 1]
                else:
                    proba[list(idxs)] = self.global_pos_rate
                continue

            proba[list(idxs)] = model.predict_proba(X[list(idxs)])[:, 1]
        return proba

    def predict(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X, X_df_for_bucket)
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)

        preds = np.zeros(len(proba))
        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            _, matched_bucket_id = self._find_model_with_backoff(bucket_id)
            if matched_bucket_id is not None and matched_bucket_id in self.bucket_thresholds:
                alpha, beta = self.bucket_thresholds.get(matched_bucket_id, (self.global_alpha, self.global_beta))
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
        return df.sort_values(by="n_samples_all", ascending=False)
