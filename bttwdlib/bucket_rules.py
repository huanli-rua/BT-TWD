import pandas as pd
from .utils_logging import log_info


class BucketTree:
    def __init__(self, levels_cfg: list, feature_names: list[str]):
        self.levels_cfg = levels_cfg
        self.feature_names = feature_names

    def _assign_single_level(self, series: pd.Series, level_cfg: dict) -> pd.Series:
        if level_cfg.get("type") == "numeric_bin":
            bins = level_cfg.get("bins", [])
            labels = level_cfg.get("labels")
            cut_bins = [-float("inf")] + bins + [float("inf")]
            if labels is None:
                labels = [f"bin_{i}" for i in range(len(cut_bins) - 1)]
            return pd.cut(series, bins=cut_bins, labels=labels, include_lowest=True)
        if level_cfg.get("type") == "categorical_group":
            mapping = {}
            groups = level_cfg.get("groups", {})
            for group_name, values in groups.items():
                for v in values:
                    mapping[v] = group_name
            return series.map(mapping).fillna("unknown")
        return pd.Series(["unknown"] * len(series), index=series.index)

    def assign_buckets(self, X_df: pd.DataFrame) -> pd.Series:
        bucket_parts = []
        for level_cfg in self.levels_cfg:
            col = level_cfg.get("col")
            part = self._assign_single_level(X_df[col], level_cfg)
            unknown_mask = part.isna()
            if unknown_mask.any():
                log_info(f"【桶树】列 {col} 出现未知取值，{unknown_mask.sum()} 条记录记为 unknown")
                part = part.astype(object).fillna("unknown")
            bucket_parts.append(part.astype(str).apply(lambda v: f"{level_cfg.get('name')}={v}"))
        bucket_id = bucket_parts[0]
        for idx in range(1, len(bucket_parts)):
            bucket_id = bucket_id + "|" + bucket_parts[idx]
        log_info(f"【桶树】已为样本生成桶ID，共 {bucket_id.nunique()} 个组合")
        return bucket_id

    def get_level_names(self) -> list[str]:
        return [lvl.get("name") for lvl in self.levels_cfg]
