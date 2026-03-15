"""生成论文 Figure 7，并导出 Table 1 统计表。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.data import load_water_quality_dataframe


def build_table1(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in frame.columns:
        series = frame[column].dropna()
        modes = series.mode()
        mode_value = modes.max() if not modes.empty else np.nan
        records.append(
            {
                "Feature": column,
                "MAX": series.max(),
                "MIN": series.min(),
                "Mean": series.mean(),
                "Median": series.median(),
                "Mode": mode_value,
                "SD": series.std(ddof=1),
            }
        )
    return pd.DataFrame(records)


def max_normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy().astype(np.float32)
    for column in normalized.columns:
        column_max = normalized[column].max()
        if pd.isna(column_max) or column_max == 0:
            continue
        normalized[column] = normalized[column] / column_max
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Figure 7 与 Table 1。")
    parser.add_argument("--data-path", default="dataset2.csv", help="输入数据路径，默认使用 dataset2.csv。")
    parser.add_argument("--figure-dir", default="repro/output/figures", help="图表输出目录。")
    parser.add_argument("--table-dir", default="repro/output/tables", help="表格输出目录。")
    args = parser.parse_args()

    figure_dir = PROJECT_ROOT / args.figure_dir
    table_dir = PROJECT_ROOT / args.table_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    data = load_water_quality_dataframe(args.data_path)
    table1 = build_table1(data)
    table1.to_csv(table_dir / "table1_statistics.csv", index=False, encoding="utf-8-sig")

    # 论文 Figure 7 的数值表现更接近“按列最大值归一化”，而不是 min-max。
    normalized_frame = max_normalize_frame(data)
    normalized_frame.head(100).to_csv(
        table_dir / "figure7_normalized_first_100.csv",
        index=False,
        encoding="utf-8-sig",
    )

    x_axis = np.arange(100)
    plt.rcParams["figure.dpi"] = 300
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(x_axis, normalized_frame.loc[:99, "PH"], color="#EC7063", s=16, label="PH")
    axes[0].scatter(
        x_axis,
        normalized_frame.loc[:99, "Temperature"],
        color="#3498DB",
        s=16,
        label="Temperature",
    )
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Normalized value")
    axes[0].legend(loc="upper right")
    axes[0].set_title("PH and Temperature")

    axes[1].scatter(
        x_axis,
        normalized_frame.loc[:99, "Conductivity"],
        color="#1ABC9C",
        s=16,
        label="Conductivity",
    )
    axes[1].scatter(
        x_axis,
        normalized_frame.loc[:99, "Turbidity"],
        color="#5D6D7E",
        s=16,
        label="Turbidity",
    )
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Normalized value")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Conductivity and Turbidity")

    fig.suptitle("Figure 7. Example time series of the first 100 time steps")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure7_covariates.png", bbox_inches="tight")
    plt.close(fig)

    print("已生成 Figure 7 与 Table 1:")
    print(figure_dir / "figure7_covariates.png")
    print(table_dir / "table1_statistics.csv")


if __name__ == "__main__":
    main()
