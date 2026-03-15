"""通过真实 batch_size × learning_rate sweep 生成论文 Figure 10。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.paper_experiments import PaperExperimentConfig, run_fm_gru_metrics, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Figure 10。")
    parser.add_argument("--batch-sizes", default="2,4,8,16,32,64,128", help="batch_size 列表。")
    parser.add_argument("--learning-rates", default="0.001,0.01,0.1", help="learning_rate 列表。")
    parser.add_argument("--repeats", type=int, default=1, help="每组参数的重复实验次数。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-path", default="dataset2.csv")
    parser.add_argument("--target-column", default="Dissolved oxygen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--teacher-prob", type=float, default=0.8)
    parser.add_argument("--fm-k", type=int, default=84)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-index-path", default="trainindex.csv")
    parser.add_argument("--test-index-path", default="testindex.csv")
    parser.add_argument("--figure-dir", default="repro/output/figures")
    parser.add_argument("--table-dir", default="repro/output/tables")
    args = parser.parse_args()

    figure_dir = PROJECT_ROOT / args.figure_dir
    table_dir = PROJECT_ROOT / args.table_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]
    learning_rates = [float(item.strip()) for item in args.learning_rates.split(",") if item.strip()]

    records = []
    rmse_grid = np.zeros((len(learning_rates), len(batch_sizes)), dtype=np.float32)
    for lr_index, learning_rate in enumerate(learning_rates):
        for batch_index, batch_size in enumerate(batch_sizes):
            config = PaperExperimentConfig(
                data_path=args.data_path,
                target_column=args.target_column,
                epochs=args.epochs,
                batch_size=batch_size,
                lr=learning_rate,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                teacher_prob=args.teacher_prob,
                fm_k=args.fm_k,
                seed=args.seed,
                train_index_path=args.train_index_path,
                test_index_path=args.test_index_path,
            )
            metrics, _ = run_fm_gru_metrics(config=config, device=device, repeats=args.repeats, use_fm=True)
            rmse_value = float(metrics["RMSE"])
            rmse_grid[lr_index, batch_index] = rmse_value
            records.append(
                {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "RMSE": rmse_value,
                }
            )

    pd.DataFrame(records).to_csv(table_dir / "figure10_rmse_grid.csv", index=False, encoding="utf-8-sig")

    x_mesh, y_mesh = np.meshgrid(np.asarray(learning_rates, dtype=np.float32), np.asarray(batch_sizes, dtype=np.int32))
    z_mesh = rmse_grid.T

    plt.rcParams["figure.dpi"] = 300
    fig = plt.figure(figsize=(8.8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.05, 0.08, 0.83, 0.82])
    ax.plot_surface(x_mesh, y_mesh, z_mesh, rstride=1, cstride=1, cmap="summer", edgecolor="none")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("batch_size")
    ax.set_zlabel("")
    ax.set_title("Figure 10. Impact of learning rate and batch size")
    fig.text(0.765, 0.56, "RMSE", rotation=0, va="center", ha="left")
    fig.subplots_adjust(left=0.03, right=0.94, bottom=0.08, top=0.90)
    fig.savefig(figure_dir / "figure10_rmse_surface.png", pad_inches=0.25)
    plt.close(fig)

    summary = {
        "source": "trained_from_scratch",
        "batch_sizes": batch_sizes,
        "learning_rates": learning_rates,
        "repeats": args.repeats,
        "device": str(device),
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "teacher_prob": args.teacher_prob,
        "fm_k": args.fm_k,
    }
    (table_dir / "figure10_rmse_grid_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("已生成 Figure 10:")
    print(figure_dir / "figure10_rmse_surface.png")
    print(table_dir / "figure10_rmse_grid.csv")


if __name__ == "__main__":
    main()
