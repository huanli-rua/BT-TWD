import numpy as np
import pandas as pd
from collections import defaultdict, deque
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from xgboost import XGBClassifier

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    _XGB_AVAILABLE = False

from .bucket_rules import get_parent_bucket_id
from .utils_logging import log_bt, log_info
from .threshold_search import search_thresholds_with_regret, compute_regret
from .bucket_gain import compute_bucket_gain, compute_bucket_score


class BTTWDModel:
    def __init__(self, cfg: dict, bucket_tree):
        self.cfg = cfg
        self.bucket_tree = bucket_tree

        bcfg = cfg.get("BTTWD", {})
        data_cfg = cfg.get("DATA", {})
        thresh_cfg = cfg.get("THRESHOLDS", {})

        self.min_bucket_size = bcfg.get("min_bucket_size", 50)
        self.max_levels = bcfg.get("max_levels", bcfg.get("max_depth", 10))
        self.min_gain_for_split = bcfg.get("min_gain_for_split", 0.0)
        self.gamma_bucket = bcfg.get("gamma_bucket", 0.0)
        self.parent_share_rate = bcfg.get("parent_share_rate", 0.0)
        self.min_parent_share = bcfg.get("min_parent_share", 0)
        self.val_ratio = bcfg.get("val_ratio", 0.2)
        self.min_val_samples_per_bucket = bcfg.get("min_val_samples_per_bucket", 10)
        self.use_global_backoff = bcfg.get("use_global_backoff", True)
        self.bucket_subsample = bcfg.get("bucket_subsample", 1.0)
        self.max_train_samples_per_bucket = bcfg.get("max_train_samples_per_bucket")
        self.score_metric = bcfg.get("score_metric", "bac_regret")
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
        self.bucket_estimator = self._build_bucket_estimator()

    def _sample_bucket_data(self, X_bucket: np.ndarray, y_bucket: np.ndarray, bucket_id: str = ""):
        """
        对桶内样本进行裁剪和随机子采样：
        1. 如果样本数 > max_train_samples_per_bucket，则随机采样上限数量；
        2. 再按 bucket_subsample 比例继续随机采样；
        """

        subsample = float(self.bucket_subsample)
        max_samples = self.max_train_samples_per_bucket

        indices = np.arange(len(y_bucket))
        if max_samples is not None and len(indices) > int(max_samples):
            indices = self.rng.choice(indices, size=int(max_samples), replace=False)

        if subsample < 1.0:
            keep = max(1, int(len(indices) * subsample))
            indices = self.rng.choice(indices, size=keep, replace=False)

        log_bt(
            f"桶{(' ' + bucket_id) if bucket_id else ''}采样：原始样本 N={len(y_bucket)} → 使用样本 n={len(indices)}"
        )

        return X_bucket[indices], y_bucket[indices]

    def _build_bucket_estimator(self, est_name=None):
        """构建桶内局部模型（例如 KNN 或逻辑回归）。"""

        bcfg = self.cfg.get("BTTWD", {})
        if est_name is None:
            est_name = bcfg.get("bucket_estimator", bcfg.get("posterior_estimator", "logreg"))
        est_name = str(est_name).lower() if est_name is not None else "logreg"

        none_aliases = {"none", "no", "null", "disabled"}
        if est_name in none_aliases:
            return None

        if est_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=bcfg.get("knn_k", 10),
                weights=bcfg.get("knn_weights", "uniform"),
                n_jobs=bcfg.get("knn_jobs", -1),
            )

        if est_name in {"logreg", "lr", "logistic", "logistic_regression"}:
            return LogisticRegression(
                max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
            )

        if est_name in {"rf", "random_forest", "randomforest"}:
            rf_cfg = bcfg.get("bucket_rf", {})
            return RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 200),
                max_depth=rf_cfg.get("max_depth", None),
                n_jobs=rf_cfg.get("n_jobs", -1),
                random_state=rf_cfg.get("random_state", 42),
            )

        if est_name in {"xgb", "xgboost"}:
            if not _XGB_AVAILABLE:
                raise RuntimeError("配置了 bucket_estimator='xgb'，但未安装 xgboost 库")
            xgb_cfg = bcfg.get("bucket_xgb", {})
            return XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 3),
                learning_rate=xgb_cfg.get("learning_rate", 0.1),
                subsample=xgb_cfg.get("subsample", 0.8),
                colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
                n_jobs=xgb_cfg.get("n_jobs", -1),
                random_state=xgb_cfg.get("random_state", 42),
            )

        if est_name in {"nb", "gnb", "naive_bayes", "naivebayes"}:
            return GaussianNB()

        log_info(f"【BTTWD】未知的 bucket_estimator='{est_name}'，回退到 logreg")
        return LogisticRegression(
            max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
        )

    def _build_global_estimator(self):
        """构建全局后验估计器（例如 XGB）。"""

        bcfg = self.cfg.get("BTTWD", {})
        est_name = bcfg.get("global_estimator", "logreg")
        est_name = str(est_name).lower() if est_name is not None else "logreg"

        none_aliases = {"none", "no", "null", "disabled"}
        if est_name in none_aliases:
            return None

        if est_name in {"xgb", "xgboost"}:
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

        if est_name in {"rf", "random_forest", "randomforest"}:
            rf_cfg = bcfg.get("global_rf", bcfg.get("bucket_rf", {}))
            return RandomForestClassifier(
                n_estimators=rf_cfg.get("n_estimators", 200),
                max_depth=rf_cfg.get("max_depth", None),
                n_jobs=rf_cfg.get("n_jobs", -1),
                random_state=rf_cfg.get("random_state", 42),
            )

        if est_name == "knn":
            return KNeighborsClassifier(
                n_neighbors=bcfg.get("knn_k", 10),
                weights=bcfg.get("knn_weights", "uniform"),
                n_jobs=bcfg.get("knn_jobs", -1),
            )

        if est_name in {"nb", "gnb", "naive_bayes", "naivebayes"}:
            return GaussianNB()

        if est_name in {"logreg", "lr", "logistic", "logistic_regression"}:
            return LogisticRegression(
                max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
            )

        log_info(f"【BTTWD】未知的 global_estimator='{est_name}'，回退到 logreg")
        return LogisticRegression(
            max_iter=bcfg.get("logreg_max_iter", 200), C=bcfg.get("logreg_C", 1.0)
        )

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
        return (self.global_alpha, self.global_beta), None

    def _init_bucket_record(self, bucket_id, parent_id, train_idx, val_idx, y):
        return {
            "bucket_id": bucket_id,
            "layer": f"L{len(bucket_id.split('|'))}",
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
        }

    def _calc_bucket_metrics(self, proba: np.ndarray, y_true: np.ndarray) -> dict:
        """
        使用当前全局阈值 alpha/beta 计算桶的结构评估指标（regret, bac）。

        注意：
        - 这里只用于“结构决策”（是否继续细分桶）；
        - 真正三支决策时，每个桶会单独进行阈值搜索，得到自己的 alpha/beta；
        - 因此这里的指标不等同于最终预测阶段使用的桶内阈值。
        """

        preds = np.where(proba >= self.global_alpha, 1, np.where(proba <= self.global_beta, 0, -1))
        regret_val = compute_regret(y_true, preds, self.costs)

        pos_mask = y_true == 1
        neg_mask = ~pos_mask
        tp = np.sum((preds == 1) & pos_mask)
        tn = np.sum((preds == 0) & neg_mask)

        tpr = tp / pos_mask.sum() if pos_mask.sum() > 0 else np.nan
        tnr = tn / neg_mask.sum() if neg_mask.sum() > 0 else np.nan
        if np.isnan(tpr) and np.isnan(tnr):
            bac = np.nan
        elif np.isnan(tpr):
            bac = tnr / 2
        elif np.isnan(tnr):
            bac = tpr / 2
        else:
            bac = 0.5 * (tpr + tnr)

        return {"regret": float(regret_val), "bac": float(bac) if not np.isnan(bac) else np.nan}

    def _split_buckets_with_gain(self, bucket_ids, proba_all: np.ndarray, y: np.ndarray):
        """
        基于桶增益判定自动决定是否继续细分。

        bucket_ids 可以是 list / ndarray / Series，函数内部统一转成 Series 处理。
        假设所有样本的 bucket_id 都来自统一的 BT 层级规则，即按 '|' 拆分后的长度在全体样本上是一致的。
        """

        bucket_ids = pd.Series(bucket_ids, dtype="string")
        parts_series = bucket_ids.str.split("|")
        if len(parts_series) == 0:
            return {}, {}

        lengths = parts_series.apply(len)
        if lengths.nunique() != 1:
            raise ValueError("bucket_id 的层级深度不一致，当前实现假设所有路径长度相同")

        num_levels = len(parts_series.iloc[0])
        level_prefixes = []
        for level in range(num_levels):
            level_prefixes.append(parts_series.apply(lambda p: "|".join(p[: level + 1])))

        level_groups = []
        for lvl_series in level_prefixes:
            level_groups.append({bid: idxs.to_numpy() for bid, idxs in lvl_series.groupby(lvl_series).groups.items()})

        leaf_index_map = {}
        visited_parent = {}
        queue = deque((bucket_id, 0) for bucket_id in level_groups[0].keys())

        while queue:
            bucket_id, level = queue.popleft()
            idx_all = level_groups[level][bucket_id]
            parent_id = get_parent_bucket_id(bucket_id)
            visited_parent[bucket_id] = parent_id

            if level + 1 >= self.max_levels or level == num_levels - 1:
                leaf_index_map[bucket_id] = idx_all
                continue

            child_level = level + 1
            child_series = level_prefixes[child_level]
            child_values = child_series.iloc[idx_all]
            child_groups = {cid: idxs.to_numpy() for cid, idxs in child_values.groupby(child_values).groups.items()}

            min_child_size = min(len(v) for v in child_groups.values()) if child_groups else 0
            if any(len(v) < self.min_bucket_size for v in child_groups.values()):
                log_bt(
                    f"桶 {bucket_id} 子桶样本不足（最小子桶 n={min_child_size} < {self.min_bucket_size}），不再细分"
                )
                leaf_index_map[bucket_id] = idx_all
                continue

            parent_metrics = self._calc_bucket_metrics(proba_all[idx_all], y[idx_all])
            parent_score = compute_bucket_score(parent_metrics, self.score_metric)
            child_scores = []
            child_weights = []
            for cid, cidx in child_groups.items():
                metrics = self._calc_bucket_metrics(proba_all[cidx], y[cidx])
                child_scores.append(compute_bucket_score(metrics, self.score_metric))
                child_weights.append(len(cidx) / len(idx_all))

            gain = compute_bucket_gain(parent_score, child_scores, child_weights, self.gamma_bucket)
            log_bt(
                f"桶 {bucket_id} 分裂前 Score={parent_score:.4f}，层级 L{level + 1}，样本 n={len(idx_all)}；子桶Score={child_scores}，Gain={gain:.4f}"
            )

            if gain < self.min_gain_for_split:
                log_bt("Gain 不足，停止在本层")
                leaf_index_map[bucket_id] = idx_all
                continue

            log_bt(f"Gain 足够，进入下一层 L{child_level + 1}")
            queue.extend((cid, child_level) for cid in child_groups.keys())

        return leaf_index_map, visited_parent

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

        self.bucket_estimator = self._build_bucket_estimator()

        inner_train_idx = np.asarray(inner_train_idx)
        inner_val_idx = np.asarray(inner_val_idx)

        self.global_pos_rate = float(np.mean(y))

        # Step 1: 预生成桶ID
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)

        self.bucket_stats = {}
        self.threshold_logs = []

        # Step 2: 训练全局模型 + 阈值
        X_train_inner = X[inner_train_idx]
        y_train_inner = y[inner_train_idx]

        self.global_model = self._build_global_estimator()
        if self.global_model is not None:
            self.global_model.fit(X_train_inner, y_train_inner)
            log_info("【BTTWD】全局模型训练完成，用于兜底预测")
        else:
            log_info("【BTTWD】global_estimator=none：仅使用全局正类比例作为概率")

        if self.optimize_thresholds and len(inner_val_idx) > 0:
            X_val_inner = X[inner_val_idx]
            y_val_inner = y[inner_val_idx]
            if self.global_model is None:
                proba_val_inner = np.full(len(y_val_inner), self.global_pos_rate)
            else:
                proba_val_inner = self.global_model.predict_proba(X_val_inner)[:, 1]
            self.global_alpha, self.global_beta, _ = self._search_thresholds(proba_val_inner, y_val_inner)

        if self.global_model is None:
            proba_all = np.full(len(y), self.global_pos_rate)
            log_bt(
                "global_estimator=none：桶增益计算将使用全局正类率，通常不会产生正的 Gain，BT 结构将基本不细分。"
            )
        else:
            proba_all = self.global_model.predict_proba(X)[:, 1]

        # Step 3: 桶增益判定，决定是否继续细分
        leaf_index_map, visited_parent = self._split_buckets_with_gain(bucket_ids, proba_all, y)

        parent_index_map = defaultdict(list)

        for bucket_id, idx_all in leaf_index_map.items():
            y_bucket = y[idx_all]
            n_bucket = len(idx_all)
            parent_id = visited_parent.get(bucket_id)

            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)
            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            self.bucket_stats[bucket_id] = {
                **self._init_bucket_record(bucket_id, parent_id, train_idx_bucket, val_idx_bucket, y),
                "n_all": int(n_bucket),
                "pos_rate_all": float(y_bucket.mean()) if n_bucket else float("nan"),
            }

            if n_bucket < self.min_bucket_size:
                self.bucket_stats[bucket_id]["use_parent_threshold"] = True
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

            if parent_id is not None:
                n_share = max(int(n_bucket * self.parent_share_rate), self.min_parent_share)
                n_share = min(n_share, n_bucket)
                if n_share > 0:
                    share_idx = self.rng.choice(idx_all, size=n_share, replace=False)
                    parent_index_map[parent_id].extend(share_idx.tolist())
                    log_info(f"【BTTWD】桶 {bucket_id} 向父桶 {parent_id} 贡献 {len(share_idx)} 个典型样本")

        # Step 4: 训练叶子桶模型（先完成叶子训练与阈值选择，再将样本贡献给父桶）
        self.bucket_models = {}
        self.bucket_thresholds = {}

        if self.bucket_estimator is None:
            log_info("【BTTWD】bucket_estimator=none：不训练桶内局部模型，仅使用全局模型概率做桶内阈值搜索")

        for bucket_id, idx_all in leaf_index_map.items():
            idx_all = np.asarray(idx_all)
            y_all = y[idx_all]

            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)

            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            y_train_bucket = y[train_idx_bucket]
            y_val_bucket = y[val_idx_bucket]

            record = self.bucket_stats[bucket_id]

            if len(y_train_bucket) < self.min_bucket_size or np.unique(y_train_bucket).size < 2:
                log_info(f"【BTTWD】叶子桶 {bucket_id} 训练样本不足或单类，使用父桶/全局阈值")
                record["use_parent_threshold"] = True
                continue

            model = None
            if self.bucket_estimator is not None:
                model = self._build_bucket_estimator()
                X_train_bucket = X[train_idx_bucket]
                X_train_bucket, y_train_bucket = self._sample_bucket_data(
                    X_train_bucket, y_train_bucket, bucket_id
                )
                if np.unique(y_train_bucket).size < 2:
                    log_info(
                        f"【BTTWD】叶子桶 {bucket_id} 采样后仅剩单类样本，使用父桶/全局阈值"
                    )
                    record["use_parent_threshold"] = True
                    continue
                model.fit(X_train_bucket, y_train_bucket)
                self.bucket_models[bucket_id] = model

            if (
                len(idx_all) < self.min_samples_for_thresholds
                or len(val_idx_bucket) < self.min_val_samples_per_bucket
                or np.unique(y_val_bucket).size < 2
            ):
                record["use_parent_threshold"] = True
                continue

            if model is None:
                if self.global_model is not None:
                    proba_val = self.global_model.predict_proba(X[val_idx_bucket])[:, 1]
                else:
                    proba_val = np.full(len(val_idx_bucket), self.global_pos_rate)
            else:
                proba_val = model.predict_proba(X[val_idx_bucket])[:, 1]
            alpha, beta, stats = self._search_thresholds(proba_val, y_val_bucket)

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
                }
            )

            self.threshold_logs.append(
                {
                    "bucket_id": bucket_id,
                    "layer": record.get("layer"),
                    "parent_bucket_id": record.get("parent_bucket_id", ""),
                    "n_train": record.get("n_train", 0),
                    "n_val": record.get("n_val", 0),
                    "pos_rate_train": record.get("pos_rate_train"),
                    "pos_rate_val": record.get("pos_rate_val"),
                    "alpha": record.get("alpha"),
                    "beta": record.get("beta"),
                    "regret_val": record.get("regret_val"),
                    "F1_val": record.get("F1_val"),
                    "Precision_val": record.get("Precision_val"),
                    "Recall_val": record.get("Recall_val"),
                    "BND_ratio_val": record.get("BND_ratio_val"),
                    "pos_coverage_val": record.get("pos_coverage_val"),
                    "threshold_n_samples": record.get("threshold_n_samples", 0),
                    "use_parent_threshold": record.get("use_parent_threshold", False),
                }
            )

        # Step 5: 训练父桶模型（在叶子贡献样本后进行）
        for parent_id, idx_list in parent_index_map.items():
            idx_all = np.array(sorted(set(idx_list)))
            y_all = y[idx_all]

            train_mask = np.isin(idx_all, inner_train_idx)
            val_mask = np.isin(idx_all, inner_val_idx)

            train_idx_bucket = idx_all[train_mask]
            val_idx_bucket = idx_all[val_mask]

            y_train_bucket = y[train_idx_bucket]
            y_val_bucket = y[val_idx_bucket]

            record = self.bucket_stats.get(parent_id)
            if record is None:
                record = self._init_bucket_record(parent_id, get_parent_bucket_id(parent_id), train_idx_bucket, val_idx_bucket, y)
                record["n_all"] = int(len(idx_all))
                record["pos_rate_all"] = float(y_all.mean()) if len(y_all) else float("nan")
                self.bucket_stats[parent_id] = record

            if len(y_train_bucket) < self.min_bucket_size or np.unique(y_train_bucket).size < 2:
                log_info(f"【BTTWD】父桶 {parent_id} 训练样本不足或单类，使用父桶/全局阈值")
                record["use_parent_threshold"] = True
                continue

            model = None
            if self.bucket_estimator is not None:
                model = self._build_bucket_estimator()
                X_train_bucket = X[train_idx_bucket]
                X_train_bucket, y_train_bucket = self._sample_bucket_data(
                    X_train_bucket, y_train_bucket, parent_id
                )
                if np.unique(y_train_bucket).size < 2:
                    log_info(
                        f"【BTTWD】父桶 {parent_id} 采样后仅剩单类样本，使用父桶/全局阈值"
                    )
                    record["use_parent_threshold"] = True
                    continue
                model.fit(X_train_bucket, y_train_bucket)
                self.bucket_models[parent_id] = model

            if (
                len(idx_all) < self.min_samples_for_thresholds
                or len(val_idx_bucket) < self.min_val_samples_per_bucket
                or np.unique(y_val_bucket).size < 2
            ):
                record["use_parent_threshold"] = True
                continue

            if model is None:
                if self.global_model is not None:
                    proba_val = self.global_model.predict_proba(X[val_idx_bucket])[:, 1]
                else:
                    proba_val = np.full(len(val_idx_bucket), self.global_pos_rate)
            else:
                proba_val = model.predict_proba(X[val_idx_bucket])[:, 1]
            alpha, beta, stats = self._search_thresholds(proba_val, y_val_bucket)

            self.bucket_thresholds[parent_id] = (alpha, beta)

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
                }
            )

            self.threshold_logs.append(
                {
                    "bucket_id": parent_id,
                    "layer": record.get("layer"),
                    "parent_bucket_id": record.get("parent_bucket_id", ""),
                    "n_train": record.get("n_train", 0),
                    "n_val": record.get("n_val", 0),
                    "pos_rate_train": record.get("pos_rate_train"),
                    "pos_rate_val": record.get("pos_rate_val"),
                    "alpha": record.get("alpha"),
                    "beta": record.get("beta"),
                    "regret_val": record.get("regret_val"),
                    "F1_val": record.get("F1_val"),
                    "Precision_val": record.get("Precision_val"),
                    "Recall_val": record.get("Recall_val"),
                    "BND_ratio_val": record.get("BND_ratio_val"),
                    "pos_coverage_val": record.get("pos_coverage_val"),
                    "threshold_n_samples": record.get("threshold_n_samples", 0),
                    "use_parent_threshold": record.get("use_parent_threshold", False),
                }
            )

        # 对未单独训练阈值的桶补充日志（继承父桶或全局阈值）
        for bucket_id, record in self.bucket_stats.items():
            if any(log["bucket_id"] == bucket_id for log in self.threshold_logs):
                continue
            record["use_parent_threshold"] = True
            (alpha, beta), parent_with_threshold = self._get_threshold_with_backoff(bucket_id)
            record["alpha"] = float(alpha)
            record["beta"] = float(beta)
            self.threshold_logs.append(
                {
                    "bucket_id": bucket_id,
                    "layer": record.get("layer"),
                    "parent_bucket_id": record.get("parent_bucket_id", ""),
                    "n_train": record.get("n_train", 0),
                    "n_val": record.get("n_val", 0),
                    "pos_rate_train": record.get("pos_rate_train"),
                    "pos_rate_val": record.get("pos_rate_val"),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "regret_val": record.get("regret_val"),
                    "F1_val": record.get("F1_val"),
                    "Precision_val": record.get("Precision_val"),
                    "Recall_val": record.get("Recall_val"),
                    "BND_ratio_val": record.get("BND_ratio_val"),
                    "pos_coverage_val": record.get("pos_coverage_val"),
                    "threshold_n_samples": record.get("threshold_n_samples", 0),
                    "use_parent_threshold": True,
                    "parent_with_threshold": parent_with_threshold if parent_with_threshold else "",
                }
            )

        log_info(
            "【BTTWD】共生成 "
            f"{bucket_ids.nunique()} 个叶子桶，其中有效桶 {len(self.bucket_models)} 个（样本数 ≥ {self.min_bucket_size}）"
        )

    def predict_proba(self, X: np.ndarray, X_df_for_bucket: pd.DataFrame) -> np.ndarray:
        bucket_ids = self.bucket_tree.assign_buckets(X_df_for_bucket)
        proba = np.zeros(len(X))

        if self.bucket_estimator is None:
            if self.global_model is not None:
                proba = self.global_model.predict_proba(X)[:, 1]
            else:
                proba.fill(self.global_pos_rate)
            return proba

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
