"""兼容旧入口：调用新的 Table 2/3/4 重算脚本。"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_step(script_name: str, extra_args: list[str]) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "repro" / "scripts" / script_name)]
    command.extend(extra_args)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="兼容旧入口，重算 Table 2/3/4。")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--teacher-prob", type=float, default=0.8)
    parser.add_argument("--fm-k", type=int, default=84)
    args = parser.parse_args()

    common_args = [
        "--repeats",
        str(args.repeats),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--hidden-size",
        str(args.hidden_size),
        "--num-layers",
        str(args.num_layers),
        "--dropout",
        str(args.dropout),
        "--teacher-prob",
        str(args.teacher_prob),
        "--fm-k",
        str(args.fm_k),
    ]

    run_step("generate_table2.py", common_args)
    run_step("generate_table3.py", common_args)
    run_step("generate_table4.py", common_args)

    print("已重算 Table 2 / Table 3 / Table 4。")


if __name__ == "__main__":
    main()
