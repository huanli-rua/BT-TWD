"""命令行入口：一键生成强异质性合成数据集。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bttwdlib.synth_data import (
    generate_synth_strong_v1,
    generate_synth_strong_v2,
    save_synth_strong_v1,
    save_synth_strong_v2,
)
from bttwdlib.utils_logging import log_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成强异质性二分类合成数据集")
    parser.add_argument("--version", type=str, choices=["v1", "v2"], default="v2", help="合成数据版本")
    parser.add_argument("--out", type=str, default=None, help="输出数据文件路径")
    parser.add_argument("--meta_out", type=str, default=None, help="元数据保存路径")
    parser.add_argument("--n", type=int, default=200000, help="样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--hetero_scale", type=float, default=1.0, help="异质性缩放系数")
    parser.add_argument("--n_groups", type=int, default=4, help="分组数量")
    parser.add_argument("--n_x", type=int, default=10, help="连续特征维度")
    parser.add_argument("--n_z", type=int, default=5, help="噪声特征维度")
    parser.add_argument("--eps_std", type=float, default=0.2, help="高斯噪声标准差 (仅 v1)")
    parser.add_argument("--target_rate", type=float, default=0.25, help="目标全局正例率")
    parser.add_argument("--k2", type=float, default=1.8, help="x2 翻转强度 (v2)")
    parser.add_argument("--k_inter", type=float, default=1.2, help="x1-x2 交互强度 (v2)")
    parser.add_argument("--sigma_noisy", type=float, default=0.75, help="噪声桶标准差 (v2)")
    parser.add_argument("--flip_noisy", type=float, default=0.25, help="噪声桶标签翻转率 (v2)")
    return parser.parse_args()


def main():
    args = parse_args()
    log_info(
        "【参数】" + ", ".join(
            [
                f"version={args.version}",
                f"n={args.n}",
                f"seed={args.seed}",
                f"hetero_scale={args.hetero_scale}",
                f"n_groups={args.n_groups}",
                f"n_x={args.n_x}",
                f"n_z={args.n_z}",
                f"eps_std={args.eps_std}",
                f"target_rate={args.target_rate}",
                f"k2={args.k2}",
                f"k_inter={args.k_inter}",
                f"sigma_noisy={args.sigma_noisy}",
                f"flip_noisy={args.flip_noisy}",
            ]
        )
    )

    version = args.version.lower()
    if version == "v2":
        out_path = args.out or "data/synth/synth_strong_v2.csv"
        meta_path = args.meta_out or "data/synth/synth_strong_v2_meta.json"
        df, meta = generate_synth_strong_v2(
            n=args.n,
            seed=args.seed,
            target_rate=args.target_rate,
            k2=args.k2,
            k_inter=args.k_inter,
            sigma_noisy=args.sigma_noisy,
            flip_noisy=args.flip_noisy,
            n_x=args.n_x,
            n_z=args.n_z,
        )
        save_synth_strong_v2(df, meta, out_path, meta_path)
    else:
        out_path = args.out or "data/synth/synth_strong_v1.csv"
        meta_path = args.meta_out or "data/synth/synth_strong_v1_meta.json"
        df, meta = generate_synth_strong_v1(
            n=args.n,
            seed=args.seed,
            hetero_scale=args.hetero_scale,
            n_groups=args.n_groups,
            n_x=args.n_x,
            n_z=args.n_z,
            eps_std=args.eps_std,
            target_rate=args.target_rate,
        )
        save_synth_strong_v1(df, meta, out_path, meta_path)

    log_info(f"【完成】数据与元数据已写入，示例配置可直接使用 {meta_path}")


if __name__ == "__main__":
    main()
