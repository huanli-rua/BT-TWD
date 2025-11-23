import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .bucket_rules import get_parent_bucket_id
from .utils_logging import log_info
from .threshold_search import search_thresholds_with_regret, compute_regret


class BTTWDModel:
    def __init__(self, cfg: dict, bucket_tree):
        self.cfg = cfg
        self.bucket_tree = bucket_tree

        bcfg = cfg.get("BTTWD", {})
        data_cfg = cfg.get("DATA", {})
        thresh_cfg = cfg.get("THRESHOLDS", {})

        self.min_bucket_size = bcfg.get("min_bucket_size", 50)
        self.parent_share_rate = bcfg.get("parent_share_rate", 0.0)
        self.min_parent_share = bcfg.get("min_parent_share", 0)
        self.parent_share_rate_L2 = bcfg.get("parent_share_rate_L2", self.parent_share_rate)
        self.min_parent_share_L2 = bcfg.get("min_parent_share_L2", self.min_parent_share)
        self.parent_share_rate_L1 = bcfg.get("parent_share_rate_L1", self.parent_share_rate)
        self.min_parent_share_L1 = bcfg.get("min_parent_share_L1", self.min_parent_share)
        self.val_ratio = bcfg.get("val_ratio", 0.2)
        self.min_val_samples_per_bucket = bcfg.get("min_val_samples_per_bucket", 10)
        self.use_global_backoff = bcfg.get("use_global_backoff", True)
        self.optimize_thresholds = True
        self.threshold_mode = thresh_cfg.get("mode", bcfg.get("thresholds_mode", "bucket_wise"))
        self.threshold_objective = thresh_cfg.get("objective", "regret")
        self.threshold_grid_alpha = thresh_cfg.get("alpha_grid", [])
        self.threshold_grid_beta = thresh_cfg.get("beta_grid", [])
        self.gap_min = thresh_cfg.get("gap_min", 0.0)
        self.costs = thresh_cfg.get(
            "costs",
            {
                "C_TP": 0.0,
                "C_TN": 0.0,
                "C_FP": 1.0,
                "C_FN": 3.0,
                "C_BP": 1.5,
                "C_BN": 0.5,
            },
        )
        self.global_alpha = thresh_cfg.get("alpha_init", 0.6)
        self.global_beta = thresh_cfg.get("beta_init", 0.2)
        self.min_samples_for_thresholds = thresh_cfg.get(
            "min_samples_for_thresholds", bcfg.get("min_val_samples_per_bucket", 10)
        )
        self.random_state = data_cfg.get("random_state", 42)

        self.bucket_models = {}
        self.bucket_thresholds = {}
        self.bucket_stats = {}
        self.threshold_logs = []
        self.global_model = None
        self.global_pos_rate = 0.5
        self.rng = np.random.default_rng(self.random_state)
        self.global_bucket_id = "L0"

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
        if self.global_bucket_id in self.bucket_models:
            return self.bucket_models[self.global_bucket_id], self.global_bucket_id
        return None, None

    def _search_thresholds(self, proba: np.ndarray, y_true: np.ndarray):
        grid_alpha = self.threshold_grid_alpha or [self.global_alpha]
        grid_beta = self.threshold_grid_beta or [self.global_beta]

        alpha, beta, stats = search_thresholds_with_regret(
            proba,
            y_true,
            alpha_grid=grid_alpha,
            beta_grid=grid_beta,
            costs=self.costs,
            gap_min=self.gap_min,
        )
        return alpha, beta, stats

    def _get_threshold_with_backoff(self, bucket_id: str):
        parts = bucket_id.split("|")
        for end in range(len(parts), 0, -1):
            candidate = "|".join(parts[:end])
            if candidate in self.bucket_thresholds:
                return self.bucket_thresholds[candidate], candidate
        if self.global_bucket_id in self.bucket_thresholds:
            return self.bucket_thresholds[self.global_bucket_id], self.global_bucket_id
        return (self.global_alpha, self.global_beta), None

    def _init_bucket_record(self, bucket_id, parent_id, train_idx, val_idx, y):
        if bucket_id == self.global_bucket_id:
            layer_name = "L0"
        else:
            layer_name = f"L{len(bucket_id.split('|'))}"
        return {
            "bucket_id": bucket_id,
            "layer": layer_name,
            "parent_bucket_id": parent_id if parent_id is not None else "",
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "pos_rate_train": float(y[train_idx].mean()) if len(train_idx) else float("nan"),
            "pos_rate_val": float(y[val_idx].mean()) if len(val_idx) else float("nan"),
            "alpha": float("nan"),
            "beta": float("nan"),
            "regret_val": float("nan"),
            "F1_val": float("nan"),
            "Precision_val": float("nan"),
            "Recall_val": float("nan"),
            "BND_ratio_val": float("nan"),
            "pos_coverage_val": float("nan"),
            "use_parent_threshold": False,
            "threshold_n_samples": 0,
            "parent_with_threshold": "",
        }

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

        # Step 1: 构建桶树索引
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        df_buckets = pd.DataFrame({"idx": np.arange(len(y)), "bucket_id": bucket_ids, "y": y})

        base_train = {}
        base_val = {}
        aggregated_train = defaultdict(list)
        aggregated_val = defaultdict(list)
        parent_children = defaultdict(list)

        self.bucket_stats = {}
        self.threshold_logs = []

        for bucket_id, g in df_buckets.groupby("bucket_id"):
            idx_all = g["idx"].to_numpy()
            y_bucket = g["y"].to_numpy()
            n_bucket = len(idx_all)

            parent_id = get_parent_bucket_id(bucket_id)
            if parent_id is not None:
                parent_children[parent_id].append(bucket_id)

            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)
            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            base_train[bucket_id] = list(train_idx_bucket)
            base_val[bucket_id] = list(val_idx_bucket)
            aggregated_train[bucket_id].extend(train_idx_bucket.tolist())
            aggregated_val[bucket_id].extend(val_idx_bucket.tolist())

            self.bucket_stats[bucket_id] = {
                **self._init_bucket_record(bucket_id, parent_id, train_idx_bucket, val_idx_bucket, y),
                "n_all": int(n_bucket),
                "pos_rate_all": float(y_bucket.mean()) if n_bucket else float("nan"),
            }

        # 确保父节点与 L0 存在记录
        for parent_id in parent_children.keys():
            if parent_id not in self.bucket_stats:
                self.bucket_stats[parent_id] = self._init_bucket_record(
                    parent_id, get_parent_bucket_id(parent_id), [], [], y
                )
                self.bucket_stats[parent_id]["n_all"] = 0
                self.bucket_stats[parent_id]["pos_rate_all"] = float("nan")
            aggregated_train[parent_id]
            aggregated_val[parent_id]

        if self.global_bucket_id not in self.bucket_stats:
            self.bucket_stats[self.global_bucket_id] = self._init_bucket_record(
                self.global_bucket_id, None, [], [], y
            )
            self.bucket_stats[self.global_bucket_id]["n_all"] = int(len(inner_train_idx))
            self.bucket_stats[self.global_bucket_id]["pos_rate_all"] = float(self.global_pos_rate)
        aggregated_train[self.global_bucket_id]
        aggregated_val[self.global_bucket_id]

        max_depth = 0
        for bucket_id in self.bucket_stats.keys():
            if bucket_id == self.global_bucket_id:
                continue
            max_depth = max(max_depth, len(bucket_id.split("|")))

        def _share_params(depth: int):
            if depth >= 3:
                return self.parent_share_rate, self.min_parent_share
            if depth == 2:
                return self.parent_share_rate_L2, self.min_parent_share_L2
            return self.parent_share_rate_L1, self.min_parent_share_L1

        # Step 2: 逐层供样：L3→L2→L1→L0
        for depth in range(max_depth, 0, -1):
            buckets_at_depth = [
                b for b in self.bucket_stats.keys() if b != self.global_bucket_id and len(b.split("|")) == depth
            ]
            share_rate, share_min = _share_params(depth)
            for bucket_id in buckets_at_depth:
                parent_id = get_parent_bucket_id(bucket_id)
                if parent_id is None and depth == 1:
                    parent_id = self.global_bucket_id
                if parent_id is None:
                    continue

                train_candidates = np.array(aggregated_train.get(bucket_id, []), dtype=int)
                val_candidates = np.array(aggregated_val.get(bucket_id, []), dtype=int)
                n_share_train = min(len(train_candidates), max(int(len(train_candidates) * share_rate), share_min))
                n_share_val = min(len(val_candidates), max(int(len(val_candidates) * share_rate), share_min))

                if n_share_train > 0:
                    share_train = self.rng.choice(train_candidates, size=n_share_train, replace=False)
                    aggregated_train[parent_id].extend(share_train.tolist())
                if n_share_val > 0:
                    share_val = self.rng.choice(val_candidates, size=n_share_val, replace=False)
                    aggregated_val[parent_id].extend(share_val.tolist())

                log_info(
                    f"【BTTWD】桶 {bucket_id} 向父桶 {parent_id} 贡献 {n_share_train} 个训练样本，{n_share_val} 个验证样本"
                )

        # Step 3: 训练全局 L0 概率模型（使用抽象代表样本）
        global_train_idx = np.array(sorted(set(aggregated_train.get(self.global_bucket_id, inner_train_idx.tolist()))))
        if len(global_train_idx) == 0:
            global_train_idx = inner_train_idx
        self.global_model = self._build_global_estimator()
        self.global_model.fit(X[global_train_idx], y[global_train_idx])
        self.bucket_models[self.global_bucket_id] = self.global_model
        log_info("【BTTWD】全局模型(L0)训练完成，用于兜底预测")

        if self.optimize_thresholds:
            global_val_idx = np.array(sorted(set(aggregated_val.get(self.global_bucket_id, inner_val_idx.tolist()))))
            if len(global_val_idx) == 0:
                global_val_idx = global_train_idx
            proba_val_inner = self.global_model.predict_proba(X[global_val_idx])[:, 1]
            self.global_alpha, self.global_beta, global_stats = self._search_thresholds(proba_val_inner, y[global_val_idx])
            self.bucket_thresholds[self.global_bucket_id] = (self.global_alpha, self.global_beta)
            record = self.bucket_stats[self.global_bucket_id]
            record.update(
                {
                    "alpha": float(self.global_alpha),
                    "beta": float(self.global_beta),
                    "regret_val": float(global_stats.get("regret", np.nan)),
                    "F1_val": float(global_stats.get("f1", np.nan)),
                    "Precision_val": float(global_stats.get("precision", np.nan)),
                    "Recall_val": float(global_stats.get("recall", np.nan)),
                    "BND_ratio_val": float(global_stats.get("bnd_ratio", np.nan)),
                    "pos_coverage_val": float(global_stats.get("pos_coverage", np.nan)),
                    "threshold_n_samples": int(global_stats.get("n_samples", 0)),
                    "use_parent_threshold": False,
                }
            )
            log_info(
                f"【BTTWD】阈值(L0): alpha={self.global_alpha:.3f}, beta={self.global_beta:.3f}, Regret={global_stats.get('regret', float('nan')):.3f}"
            )

        # Step 4: 训练各层桶模型与阈值
        self.bucket_models = {}

        train_order = sorted(
            [b for b in aggregated_train.keys() if b != self.global_bucket_id],
            key=lambda x: len(x.split("|")),
            reverse=True,
        )

        for bucket_id in train_order:
            train_idx = np.array(sorted(set(aggregated_train.get(bucket_id, []))), dtype=int)
            val_idx = np.array(sorted(set(aggregated_val.get(bucket_id, []))), dtype=int)

            parent_id = get_parent_bucket_id(bucket_id)
            if parent_id is None and len(bucket_id.split("|")) == 1:
                parent_id = self.global_bucket_id

            if bucket_id not in self.bucket_stats:
                self.bucket_stats[bucket_id] = self._init_bucket_record(bucket_id, parent_id, train_idx, val_idx, y)
                self.bucket_stats[bucket_id]["n_all"] = int(len(train_idx) + len(val_idx))
                self.bucket_stats[bucket_id]["pos_rate_all"] = float(
                    y[np.concatenate([train_idx, val_idx])].mean() if len(train_idx) + len(val_idx) else float("nan")
                )

            record = self.bucket_stats[bucket_id]
            record["n_train"] = int(len(train_idx))
            record["n_val"] = int(len(val_idx))
            record["pos_rate_train"] = float(y[train_idx].mean()) if len(train_idx) else float("nan")
            record["pos_rate_val"] = float(y[val_idx].mean()) if len(val_idx) else float("nan")

            effective_min = 2 if bucket_id == self.global_bucket_id else self.min_bucket_size
            if len(train_idx) < effective_min or np.unique(y[train_idx]).size < 2:
                log_info(f"【BTTWD】桶 {bucket_id} 训练样本不足或单类，跳过局部模型训练")
                record["use_parent_threshold"] = True
                continue

            model = self._build_bucket_estimator()
            model.fit(X[train_idx], y[train_idx])

            val_for_threshold = val_idx if len(val_idx) else train_idx
            y_val_bucket = y[val_for_threshold]
            proba_val = model.predict_proba(X[val_for_threshold])[:, 1]
            alpha, beta, stats = self._search_thresholds(proba_val, y_val_bucket)

            self.bucket_models[bucket_id] = model
            self.bucket_thresholds[bucket_id] = (alpha, beta)

            record.update(
                {
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "regret_val": float(stats.get("regret", np.nan)),
                    "F1_val": float(stats.get("f1", np.nan)),
                    "Precision_val": float(stats.get("precision", np.nan)),
                    "Recall_val": float(stats.get("recall", np.nan)),
                    "BND_ratio_val": float(stats.get("bnd_ratio", np.nan)),
                    "pos_coverage_val": float(stats.get("pos_coverage", np.nan)),
                    "threshold_n_samples": int(stats.get("n_samples", 0)),
                    "use_parent_threshold": False,
                }
            )

            log_info(
                f"【BTTWD】阈值({bucket_id}): alpha={alpha:.3f}, beta={beta:.3f}, Regret={stats.get('regret', float('nan')):.3f}"
            )

        # Step 5: 补全阈值日志与回退链
        self.threshold_logs = []
        for bucket_id, record in self.bucket_stats.items():
            if bucket_id in self.bucket_thresholds:
                record["use_parent_threshold"] = False
                record.setdefault("parent_with_threshold", "")
            else:
                (alpha, beta), parent_with_threshold = self._get_threshold_with_backoff(bucket_id)
                record["alpha"] = float(alpha)
                record["beta"] = float(beta)
                record["use_parent_threshold"] = True
                record["parent_with_threshold"] = parent_with_threshold if parent_with_threshold else ""

            total_indices = np.array(sorted(set(aggregated_train.get(bucket_id, []) + aggregated_val.get(bucket_id, []))))
            record["n_all"] = int(len(total_indices))
            record["pos_rate_all"] = float(y[total_indices].mean()) if len(total_indices) else record.get("pos_rate_all", float("nan"))

            self.threshold_logs.append(
                {
                    "bucket_id": bucket_id,
                    "layer": record.get("layer"),
                    "parent_bucket_id": record.get("parent_bucket_id", ""),
                    "n_train": record.get("n_train", 0),
                    "n_val": record.get("n_val", 0),
                    "pos_rate_train": record.get("pos_rate_train"),
                    "pos_rate_val": record.get("pos_rate_val"),
                    "alpha": record.get("alpha", float("nan")),
                    "beta": record.get("beta", float("nan")),
                    "regret_val": record.get("regret_val"),
                    "F1_val": record.get("F1_val"),
                    "Precision_val": record.get("Precision_val"),
                    "Recall_val": record.get("Recall_val"),
                    "BND_ratio_val": record.get("BND_ratio_val"),
                    "pos_coverage_val": record.get("pos_coverage_val"),
                    "threshold_n_samples": record.get("threshold_n_samples", 0),
                    "use_parent_threshold": record.get("use_parent_threshold", False),
                    "parent_with_threshold": record.get("parent_with_threshold", ""),
                }
            )

        log_info(
            f"【BTTWD】共生成 {bucket_ids.nunique()} 个叶子桶，其中有效桶 {len(self.bucket_models)} 个（样本数 ≥ {self.min_bucket_size}）"
        )

    def predict_proba(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        proba = np.zeros(len(X))

        for bucket_id, idxs in bucket_ids.groupby(bucket_ids).groups.items():
            model, _ = self._find_model_with_backoff(bucket_id)

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
            (alpha, beta), _ = self._get_threshold_with_backoff(bucket_id)

            bucket_proba = proba[list(idxs)]
            bucket_pred = np.where(bucket_proba >= alpha, 1, np.where(bucket_proba <= beta, 0, -1))
            preds[list(idxs)] = bucket_pred
        return preds

    def get_bucket_stats(self) -> pd.DataFrame:
        if not self.bucket_stats:
            return pd.DataFrame()
        df = pd.DataFrame(self.bucket_stats.values())
        sort_col = "n_all" if "n_all" in df.columns else None
        if sort_col:
            df = df.sort_values(by=sort_col, ascending=False)
        return df

    def get_threshold_logs(self) -> pd.DataFrame:
        if not self.threshold_logs:
            return pd.DataFrame()
        return pd.DataFrame(self.threshold_logs)
