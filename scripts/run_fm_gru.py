"""训练清理后的 FM-GRU 主线，并导出指标和拟合曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.data import DatasetConfig, FMGRUDataset, build_train_test_split
from repro.src.fm_gru_model import FMGRUModel
from repro.src.metrics import average_metric_dicts, compute_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def read_index_file(path_like: str | Path) -> list[int]:
    frame = pd.read_csv(PROJECT_ROOT / path_like)
    return frame.iloc[:, 0].astype(int).tolist()


def build_split(
    dataset: FMGRUDataset,
    train_ratio: float,
    seed: int,
    train_index_path: str | None,
    test_index_path: str | None,
):
    if train_index_path:
        train_indices = read_index_file(train_index_path)
        if test_index_path:
            test_indices = read_index_file(test_index_path)
        else:
            train_index_set = set(train_indices)
            test_indices = [index for index in range(len(dataset)) if index not in train_index_set]
        return (
            Subset(dataset, train_indices),
            Subset(dataset, test_indices),
            {
                "split_mode": "fixed_indices",
                "train_index_path": train_index_path,
                "test_index_path": test_index_path,
                "train_size": len(train_indices),
                "test_size": len(test_indices),
            },
        )

    trainset, testset = build_train_test_split(dataset, train_ratio, seed)
    return (
        trainset,
        testset,
        {
            "split_mode": "random_split",
            "train_ratio": train_ratio,
            "train_size": len(trainset),
            "test_size": len(testset),
        },
    )


def save_fit_plot(output_path: Path, true_values: np.ndarray, pred_values: np.ndarray, title: str) -> None:
    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(true_values, label="True value", color="#27AE60", alpha=0.35)
    ax.plot(pred_values, label="Predicted value", color="#27AE60")
    ax.set_xlabel("time step")
    ax.set_ylabel("DO mg/l")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 FM-GRU 并导出复现结果。")
    parser.add_argument("--data-path", default="KRMData/oxy/原始数据平滑训练和验证集不平滑测试集.csv")
    parser.add_argument("--target-column", default="Dissolved oxygen")
    parser.add_argument("--encode-step", type=int, default=24)
    parser.add_argument("--forecast-step", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--teacher-prob", type=float, default=0.8)
    parser.add_argument("--fm-k", type=int, default=84)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--train-index-path", default=None, help="固定训练集索引文件，可选。")
    parser.add_argument("--test-index-path", default=None, help="固定测试集索引文件，可选。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default="repro/output/runs/fm_gru_clean")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        data_path=args.data_path,
        encode_step=args.encode_step,
        forecast_step=args.forecast_step,
        target_column=args.target_column,
    )
    dataset = FMGRUDataset(config)
    trainset, testset, split_info = build_split(
        dataset=dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
        train_index_path=args.train_index_path,
        test_index_path=args.test_index_path,
    )

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = FMGRUModel(
        target_size=1,
        feature_size=4,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        forecast_step=args.forecast_step,
        encode_step=args.encode_step,
        teacher_prob=args.teacher_prob,
        fm_k=args.fm_k,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_func = torch.nn.MSELoss()

    train_losses: list[dict[str, float]] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for hisx, hisz, futx, z in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            hisx = hisx.to(device)
            hisz = hisz.to(device)
            futx = futx.to(device)
            z = z.to(device)

            optimizer.zero_grad()
            zhat_all, _, _ = model(hisx, hisz, futx, z)
            zhat_all = zhat_all.reshape(z.shape)
            loss = loss_func(zhat_all, z)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().item())
            batch_count += 1

        train_losses.append({"epoch": epoch, "loss": running_loss / max(batch_count, 1)})

    model.eval()
    batch_metrics: list[dict[str, float]] = []
    pooled_preds: list[np.ndarray] = []
    pooled_trues: list[np.ndarray] = []
    last_batch_true: np.ndarray | None = None
    last_batch_pred: np.ndarray | None = None

    with torch.no_grad():
        for hisx, hisz, futx, z in test_loader:
            hisx = hisx.to(device)
            hisz = hisz.to(device)
            futx = futx.to(device)
            z = z.to(device)

            _, zhat, _ = model(hisx, hisz, futx, z)
            z = z[:, -args.forecast_step :, :]
            zhat = zhat.reshape(z.shape)

            pred_original = dataset.inverse_target_transform(zhat.detach().cpu().numpy())
            true_original = dataset.inverse_target_transform(z.detach().cpu().numpy())

            batch_metrics.append(compute_metrics(pred_original, true_original))
            pooled_preds.append(pred_original.reshape(-1))
            pooled_trues.append(true_original.reshape(-1))
            last_batch_true = true_original.reshape(-1)
            last_batch_pred = pred_original.reshape(-1)

    batch_average = average_metric_dicts(batch_metrics)
    pooled = compute_metrics(np.concatenate(pooled_preds), np.concatenate(pooled_trues))
    elapsed = time.time() - start_time

    pd.DataFrame(train_losses).to_csv(output_dir / "train_loss.csv", index=False, encoding="utf-8-sig")

    metrics_frame = pd.DataFrame(
        [
            {
                "metric": metric_name,
                "batch_average": batch_average.get(metric_name),
                "pooled": pooled.get(metric_name),
            }
            for metric_name in ["MAE", "MSE", "RMSE", "NRMSE", "R"]
        ]
    )
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_path": args.data_path,
        "target_column": args.target_column,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "teacher_prob": args.teacher_prob,
        "fm_k": args.fm_k,
        "seed": args.seed,
        "device": str(device),
        "train_ratio": args.train_ratio,
        "split": split_info,
        "elapsed_seconds": elapsed,
        "batch_average_metrics": batch_average,
        "pooled_metrics": pooled,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if last_batch_true is not None and last_batch_pred is not None:
        pd.DataFrame(
            {
                "time_step": list(range(len(last_batch_true))),
                "true_value": last_batch_true,
                "predicted_value": last_batch_pred,
            }
        ).to_csv(output_dir / "last_batch_predictions.csv", index=False, encoding="utf-8-sig")
        save_fit_plot(
            output_dir / "figure8_like_last_batch.png",
            last_batch_true,
            last_batch_pred,
            "FM-GRU fit on the last batch of the test set",
        )

    torch.save(model.state_dict(), output_dir / "model_state.pt")

    print("训练完成，结果目录:")
    print(output_dir)
    print("batch_average_metrics =", json.dumps(batch_average, ensure_ascii=False))
    print("pooled_metrics =", json.dumps(pooled, ensure_ascii=False))
    print(f"time_consumption = {elapsed:.2f}s")


if __name__ == "__main__":
    main()
