"""基于真实消融实验生成论文 Table 3。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.paper_experiments import PaperExperimentConfig, run_fm_gru_metrics, select_device


def main() -> None:
    parser = argparse.ArgumentParser(description="训练并生成 Table 3。")
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
    args = parser.parse_args()

    table_dir = PROJECT_ROOT / args.table_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
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

    baseline_metrics, split_info = run_fm_gru_metrics(config=config, device=device, repeats=args.repeats, use_fm=False)
    fm_gru_metrics, _ = run_fm_gru_metrics(config=config, device=device, repeats=args.repeats, use_fm=True)

    frame = pd.DataFrame(
        [
            {"Model": "Baseline Model", **{key: baseline_metrics[key] for key in ["MAE", "MSE", "RMSE", "NRMSE"]}},
            {"Model": "FM-GRU", **{key: fm_gru_metrics[key] for key in ["MAE", "MSE", "RMSE", "NRMSE"]}},
        ]
    )
    frame.to_csv(table_dir / "table3_ablation.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source": "trained_from_scratch",
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
    (table_dir / "table3_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("已生成 Table 3:")
    print(table_dir / "table3_ablation.csv")


if __name__ == "__main__":
    main()
