"""导出历史 FM-GRU 拟合快照，仅用于和训练版 Figure 8 做对照。"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def extract_series(script_path: Path) -> tuple[list[float], list[float]]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    values: dict[str, list[float]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"pre", "true"}:
            values[target.id] = ast.literal_eval(node.value)
    if "true" not in values or "pre" not in values:
        raise ValueError("未能从 modelFit.py 中解析出 pre/true 数组。")
    return values["true"], values["pre"]


def main() -> None:
    parser = argparse.ArgumentParser(description="导出历史 FM-GRU 拟合快照。")
    parser.add_argument("--script-path", default="dataAnalysis/modelFit.py", help="旧脚本路径。")
    parser.add_argument("--figure-dir", default="repro/output/figures", help="图表输出目录。")
    parser.add_argument("--table-dir", default="repro/output/tables", help="表格输出目录。")
    args = parser.parse_args()

    figure_dir = PROJECT_ROOT / args.figure_dir
    table_dir = PROJECT_ROOT / args.table_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    true_values, pred_values = extract_series(PROJECT_ROOT / args.script_path)
    frame = pd.DataFrame(
        {
            "time_step": list(range(len(true_values))),
            "true_value": true_values,
            "predicted_value": pred_values,
        }
    )
    table_path = table_dir / "historical_reference_figure8_two_line.csv"
    figure_path = figure_dir / "historical_reference_figure8_two_line.png"
    frame.to_csv(table_path, index=False, encoding="utf-8-sig")

    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frame["time_step"], frame["true_value"], label="True value", color="#27AE60", alpha=0.35)
    ax.plot(frame["time_step"], frame["predicted_value"], label="Predicted value", color="#27AE60")
    ax.set_xlabel("time step")
    ax.set_ylabel("DO mg/l")
    ax.set_title("Historical FM-GRU fit snapshot from dataAnalysis/modelFit.py")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    print("已导出历史 FM-GRU 双线快照，仅供对照，不是论文主 Figure 8:")
    print(figure_path)
    print(table_path)


if __name__ == "__main__":
    main()
