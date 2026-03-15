"""一键执行论文复现流程，默认全部走真实重算/重训链路。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_step(script_name: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "repro" / "scripts" / script_name)]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="一键执行论文复现流程。")
    parser.add_argument("--figure8-epochs", type=int, default=10, help="Figure 8 训练轮数。")
    parser.add_argument("--figure8-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--include-training", action="store_true", help="是否追加 FM-GRU 训练。")
    parser.add_argument("--include-tables", action="store_true", help="是否生成 Table 2/3/4。")
    parser.add_argument("--include-figure9", action="store_true", help="是否生成 Figure 9。")
    parser.add_argument("--include-figure10", action="store_true", help="是否生成 Figure 10。")
    parser.add_argument("--table-repeats", type=int, default=10, help="表格重算重复次数。")
    parser.add_argument("--sweep-repeats", type=int, default=1, help="Figure 9/10 sweep 重复次数。")
    parser.add_argument("--training-epochs", type=int, default=10, help="训练轮数。")
    parser.add_argument("--training-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--training-output-dir", default="repro/output/runs/fm_gru_clean")
    parser.add_argument(
        "--include-archived-reference",
        action="store_true",
        help="是否额外导出历史双线快照，仅供参考，不属于从头训练结果。",
    )
    args = parser.parse_args()

    run_step("generate_figure7.py")
    run_step(
        "generate_figure8.py",
        ["--epochs", str(args.figure8_epochs), "--device", args.figure8_device],
    )
    if args.include_tables:
        run_step(
            "export_paper_tables.py",
            ["--repeats", str(args.table_repeats), "--device", args.figure8_device],
        )
    if args.include_figure9:
        run_step(
            "generate_figure9.py",
            ["--repeats", str(args.sweep_repeats), "--device", args.figure8_device],
        )
    if args.include_figure10:
        run_step(
            "generate_figure10.py",
            ["--repeats", str(args.sweep_repeats), "--device", args.figure8_device],
        )

    if args.include_archived_reference:
        run_step("export_archived_figure8.py")

    if args.include_training:
        run_step(
            "run_fm_gru.py",
            [
                "--epochs",
                str(args.training_epochs),
                "--device",
                args.training_device,
                "--output-dir",
                args.training_output_dir,
            ],
        )

    print("论文复现脚本已执行完成。")


if __name__ == "__main__":
    main()
