"""通用入口：读取指定配置跑一次分层切分 + BTTWD 训练/评估。

默认使用仓库内的 airlines 延误示例配置，可通过命令行参数替换为任何数据集配置。
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bttwdlib import (  # noqa: E402
    load_dataset,
    load_yaml_cfg,
    prepare_features_and_labels,
    run_holdout_experiment,
    run_kfold_experiments,
    show_cfg,
)
from bttwdlib.utils_logging import log_info  # noqa: E402
from bttwdlib.utils_seed import set_global_seed  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "airlines_delay.yaml"


def _build_bucket_feature_df(df_raw, cfg) -> tuple[np.ndarray, np.ndarray, object, object, object]:
    X, y, meta = prepare_features_and_labels(df_raw, cfg)
    prep_cfg = cfg.get("PREPROCESS", {})
    bucket_cols: List[str] = (prep_cfg.get("continuous_cols") or []) + (prep_cfg.get("categorical_cols") or [])
    bucket_df = df_raw[bucket_cols].reset_index(drop=True)
    return X, y, meta, bucket_df, bucket_cols


def parse_args():
    parser = argparse.ArgumentParser(description="Run BTTWD training & eval with a YAML config.")
    parser.add_argument(
        "--config",
        "--cfg",
        default=str(DEFAULT_CONFIG_PATH),
        help=(
            "Path to the YAML configuration file. "
            "Defaults to the sample airlines delay config shipped with the repo."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = load_yaml_cfg(cfg_path)
    show_cfg(cfg)
    set_global_seed(cfg.get("SEED", {}).get("global_seed", 42))

    df_raw, target_col = load_dataset(cfg)
    data_cfg = cfg.get("DATA", {})
    log_info(
        f"【入口】数据集={data_cfg.get('dataset_name')}，样本数={len(df_raw)}，标签列={target_col}"
    )

    bucket_levels = cfg.get("BTTWD", {}).get("bucket_levels", [])
    log_info(f"【桶树层级】分裂顺序={[lvl.get('name') for lvl in bucket_levels]}")

    X, y, meta, bucket_df, bucket_cols = _build_bucket_feature_df(df_raw, cfg)

    use_kfold = data_cfg.get("use_kfold", False)
    if isinstance(use_kfold, str):
        use_kfold = use_kfold.strip().lower() in {"1", "true", "yes", "y"}
    if use_kfold:
        log_info("【模式选择】use_kfold=true，启动K折实验...")
        run_kfold_experiments(X, y, bucket_df, cfg)
        return

    run_holdout_experiment(X, y, bucket_df, cfg, bucket_cols=bucket_cols)


if __name__ == "__main__":
    main()
