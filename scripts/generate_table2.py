"""基于真实训练/评估生成论文 Table 2。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.paper_experiments import PaperExperimentConfig, run_model_suite, select_device


def plot_table2(frame, output_path: Path) -> None:
    metrics = ["MAE", "MSE", "RMSE", "NRMSE"]
    methods = frame["Model"].tolist()
    x = np.arange(len(methods))
    width = 0.18

    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2E86C1", "#E74C3C", "#27AE60", "#566573"]
    for index, metric in enumerate(metrics):
        ax.bar(x + width * (index - 1.5), frame[metric], width=width, label=metric, color=colors[index])

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Metric value")
    ax.set_title("Table 2. Recomputed comparison on dissolved oxygen")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并生成 Table 2。")
    parser.add_argument(
        "--models",
        default="HA,ARIMA,LR,XGBoost,FFNN,FC-LSTM,FC-GRU,FM-GRU",
        help="参与对比的模型列表，逗号分隔。",
    )
    parser.add_argument("--repeats", type=int, default=10, help="重复实验次数。")
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
    parser.add_argument("--fm-k", type=int, default=84)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-index-path", default="trainindex.csv")
    parser.add_argument("--test-index-path", default="testindex.csv")
    parser.add_argument("--table-dir", default="repro/output/tables")
    parser.add_argument("--figure-dir", default="repro/output/figures")
    args = parser.parse_args()

    table_dir = PROJECT_ROOT / args.table_dir
    figure_dir = PROJECT_ROOT / args.figure_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

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
        fm_k=args.fm_k,
        seed=args.seed,
        train_index_path=args.train_index_path,
        test_index_path=args.test_index_path,
    )
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    device = select_device(args.device)

    frame, split_info = run_model_suite(config=config, device=device, models=models, repeats=args.repeats)
    frame.to_csv(table_dir / "table2_do_comparison.csv", index=False, encoding="utf-8-sig")
    plot_table2(frame, figure_dir / "table2_comparison.png")

    summary = {
        "source": "trained_from_scratch",
        "target_column": args.target_column,
        "models": models,
        "repeats": args.repeats,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "teacher_prob": args.teacher_prob,
        "fm_k": args.fm_k,
        "split": split_info,
    }
    (table_dir / "table2_do_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("已生成 Table 2:")
    print(table_dir / "table2_do_comparison.csv")
    print(figure_dir / "table2_comparison.png")


if __name__ == "__main__":
    main()
