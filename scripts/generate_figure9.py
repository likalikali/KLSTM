"""通过真实 K 参数 sweep 生成论文 Figure 9。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.paper_experiments import PaperExperimentConfig, run_fm_gru_metrics, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Figure 9。")
    parser.add_argument("--k-values", default="4,12,24,36,48,60,72,84,96,108,120", help="K 取值列表。")
    parser.add_argument("--repeats", type=int, default=1, help="每个 K 的重复实验次数。")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-path", default="dataset2.csv")
    parser.add_argument("--target-column", default="Dissolved oxygen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--teacher-prob", type=float, default=0.8)
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
    k_values = [int(item.strip()) for item in args.k_values.split(",") if item.strip()]
    records = []

    for k_value in k_values:
        config = PaperExperimentConfig(
            data_path=args.data_path,
            target_column=args.target_column,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            teacher_prob=args.teacher_prob,
            fm_k=k_value,
            seed=args.seed,
            train_index_path=args.train_index_path,
            test_index_path=args.test_index_path,
        )
        metrics, _ = run_fm_gru_metrics(config=config, device=device, repeats=args.repeats, use_fm=True)
        records.append({"K": k_value, **{name: metrics[name] for name in ["MAE", "MSE", "RMSE", "NRMSE"]}})

    frame = pd.DataFrame(records)
    frame.to_csv(table_dir / "figure9_k_metrics.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(figsize=(9, 5))
    series = [
        ("MAE", "#EC7063"),
        ("MSE", "#48C9B0"),
        ("RMSE", "#5DADE2"),
        ("NRMSE", "#707B7C"),
    ]
    for metric, color in series:
        ax.plot(frame["K"], frame[metric], marker="o", color=color, label=metric, linewidth=2)

    ax.set_xlabel("K")
    ax.set_ylabel("Metric value")
    ax.set_title("Figure 9. Impact of parameter K on four metrics")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "figure9_k_impact.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source": "trained_from_scratch",
        "k_values": k_values,
        "repeats": args.repeats,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "teacher_prob": args.teacher_prob,
    }
    (table_dir / "figure9_k_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("已生成 Figure 9:")
    print(figure_dir / "figure9_k_impact.png")
    print(table_dir / "figure9_k_metrics.csv")


if __name__ == "__main__":
    main()
