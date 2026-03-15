"""从头训练 FM-GRU 和 FC-LSTM，生成论文 Figure 8 的三线对比图。"""

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
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repro.src.data import DatasetConfig, FMGRUDataset, resolve_project_path
from repro.src.fc_lstm_model import FCLSTMPointModel
from repro.src.fm_gru_model import FMGRUModel

LEGACY_FC_FEATURE_COLUMNS = ["Temperature", "PH", "Conductivity", "Turbidity"]
LEGACY_FC_WINDOW_CANDIDATES = ["xxx.csv"]


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


def read_csv_with_optional_header(path_like: str | Path, header: int | None) -> pd.DataFrame:
    path = resolve_project_path(path_like)
    encodings = ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin1")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, header=header)
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, header=header)


def resolve_legacy_fc_training_path(path_like: str | None) -> Path:
    if path_like:
        return resolve_project_path(path_like)

    candidates = [
        PROJECT_ROOT / "海门湾label=4(0).csv",
        PROJECT_ROOT / "dataset2.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到 FC-LSTM 旧链路训练数据，请通过 --fc-train-data-path 指定。")


def resolve_legacy_fc_window_path(path_like: str | None) -> Path | None:
    if path_like:
        path = resolve_project_path(path_like)
        return path if path.exists() else None

    for candidate_name in LEGACY_FC_WINDOW_CANDIDATES:
        candidate = PROJECT_ROOT / candidate_name
        if candidate.exists():
            return candidate
    return None


def legacy_minmax_scale_columns(values: np.ndarray) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float32).copy()
    for column_index in range(scaled.shape[1]):
        column_min = float(np.nanmin(scaled[:, column_index]))
        column_max = float(np.nanmax(scaled[:, column_index]))
        denominator = column_max - column_min
        if denominator == 0:
            denominator = 1.0
        scaled[:, column_index] = (scaled[:, column_index] - column_min) / denominator
    return scaled


def train_fc_lstm_legacy(
    training_path: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    device: torch.device,
) -> tuple[FCLSTMPointModel, Path, int]:
    resolved_path = resolve_legacy_fc_training_path(training_path)
    frame = read_csv_with_optional_header(resolved_path, header=None)
    rowdata = np.asarray(frame.ffill(), dtype=np.float32)
    rowdata = legacy_minmax_scale_columns(rowdata)
    train_size = int(0.8 * len(rowdata))

    train_x = torch.from_numpy(rowdata[:train_size, :4]).unsqueeze(-1)
    train_y = torch.from_numpy(rowdata[:train_size, 4]).unsqueeze(-1)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = FCLSTMPointModel(hidden_size=hidden_size, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        for feature_batch, target_batch in tqdm(train_loader, desc=f"FC-LSTM {epoch}/{epochs}", leave=False):
            feature_batch = feature_batch.to(device)
            target_batch = target_batch.to(device)
            prediction = model(feature_batch)
            loss = loss_func(prediction, target_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, resolved_path, train_size


def load_legacy_fc_window(path: str | Path) -> np.ndarray:
    frame = read_csv_with_optional_header(path, header=0)
    values = np.asarray(frame.iloc[:, :4], dtype=np.float32)
    return values


def build_legacy_fc_inputs(dataset: FMGRUDataset, sample_indices: list[int]) -> np.ndarray:
    raw = dataset.raw_dataframe
    feature_maxs = raw[LEGACY_FC_FEATURE_COLUMNS].max().to_numpy(dtype=np.float32)
    feature_maxs[feature_maxs == 0] = 1.0
    blocks = []
    for sample_index in sample_indices:
        future_start = sample_index + dataset.encode_step + 2
        future_end = future_start + dataset.forecast_step
        block = raw.iloc[future_start:future_end][LEGACY_FC_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        blocks.append(block / feature_maxs)
    return np.vstack(blocks)


def predict_fc_lstm_legacy(
    model: FCLSTMPointModel,
    feature_rows: np.ndarray,
    target_max: float,
    device: torch.device,
) -> np.ndarray:
    inputs = torch.from_numpy(np.asarray(feature_rows, dtype=np.float32)).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(inputs).detach().cpu().numpy().reshape(-1)
    return prediction * target_max


def train_fm_gru(
    dataset: FMGRUDataset,
    train_indices: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    teacher_prob: float,
    fm_k: int,
    device: torch.device,
) -> FMGRUModel:
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=False, drop_last=True)
    model = FMGRUModel(
        target_size=1,
        feature_size=4,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        forecast_step=dataset.forecast_step,
        encode_step=dataset.encode_step,
        teacher_prob=teacher_prob,
        fm_k=fm_k,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for hisx, hisz, futx, z in tqdm(train_loader, desc=f"FM-GRU {epoch}/{epochs}", leave=False):
            hisx = hisx.to(device)
            hisz = hisz.to(device)
            futx = futx.to(device)
            z = z.to(device)

            prediction_all, _, _ = model(hisx, hisz, futx, z)
            prediction_all = prediction_all.reshape(z.shape)
            loss = loss_func(prediction_all, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def get_plot_block_from_sample(dataset: FMGRUDataset, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    hisx, hisz, futx, z = dataset[sample_index]
    target_block = dataset.inverse_target_transform(z[-dataset.forecast_step :, :].numpy()).reshape(-1)

    raw = dataset.raw_dataframe
    start_row = sample_index + dataset.encode_step + 2
    end_row = start_row + dataset.forecast_step
    feature_block = raw.iloc[start_row:end_row][LEGACY_FC_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    return target_block, feature_block


def predict_fm_gru_blocks(
    model: FMGRUModel,
    dataset: FMGRUDataset,
    sample_indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    predictions = []

    model.eval()
    with torch.no_grad():
        for sample_index in sample_indices:
            hisx, hisz, futx, z = dataset[sample_index]
            batch_hisx = hisx.unsqueeze(0).to(device)
            batch_hisz = hisz.unsqueeze(0).to(device)
            batch_futx = futx.unsqueeze(0).to(device)
            batch_z = z.unsqueeze(0).to(device)

            _, forecast, _ = model(batch_hisx, batch_hisz, batch_futx, batch_z)
            prediction_block = dataset.inverse_target_transform(forecast.detach().cpu().numpy()).reshape(-1)
            target_block, _ = get_plot_block_from_sample(dataset, sample_index)

            targets.append(target_block)
            predictions.append(prediction_block)

    return np.concatenate(targets), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(description="从头训练生成 Figure 8。")
    parser.add_argument("--data-path", default="dataset2.csv")
    parser.add_argument("--train-index-path", default="trainindex.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fm-batch-size", type=int, default=4)
    parser.add_argument("--fc-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--fm-num-layers", type=int, default=2)
    parser.add_argument("--fc-num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--teacher-prob", type=float, default=0.8)
    parser.add_argument("--fm-k", type=int, default=84)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fc-train-data-path", default=None, help="FC-LSTM 旧链路训练数据路径，默认自动探测。")
    parser.add_argument(
        "--fc-window-path",
        default=None,
        help="FC-LSTM 旧 Figure 8 输入窗口路径，默认自动探测根目录下的 xxx.csv。",
    )
    parser.add_argument("--figure-dir", default="repro/output/figures")
    parser.add_argument("--table-dir", default="repro/output/tables")
    parser.add_argument(
        "--plot-sample-indices",
        default="12,548,140,325,467",
        help="用于拼接 Figure 8 的 5 个样本索引，按逗号分隔。",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    figure_dir = PROJECT_ROOT / args.figure_dir
    table_dir = PROJECT_ROOT / args.table_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    dataset = FMGRUDataset(DatasetConfig(data_path=args.data_path, encode_step=24, forecast_step=12))
    train_indices = read_index_file(args.train_index_path)
    plot_sample_indices = [int(item.strip()) for item in args.plot_sample_indices.split(",") if item.strip()]

    start_time = time.time()
    fm_gru = train_fm_gru(
        dataset=dataset,
        train_indices=train_indices,
        epochs=args.epochs,
        batch_size=args.fm_batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.fm_num_layers,
        dropout=args.dropout,
        teacher_prob=args.teacher_prob,
        fm_k=args.fm_k,
        device=device,
    )
    target_values, fm_gru_values = predict_fm_gru_blocks(
        model=fm_gru,
        dataset=dataset,
        sample_indices=plot_sample_indices,
        device=device,
    )

    fc_lstm, fc_training_path, fc_train_size = train_fc_lstm_legacy(
        training_path=args.fc_train_data_path,
        epochs=args.epochs,
        batch_size=args.fc_batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.fc_num_layers,
        device=device,
    )
    fc_window_path = resolve_legacy_fc_window_path(args.fc_window_path)
    if fc_window_path is not None:
        legacy_fc_inputs = load_legacy_fc_window(fc_window_path)
    else:
        legacy_fc_inputs = build_legacy_fc_inputs(dataset, plot_sample_indices)
    if len(legacy_fc_inputs) != len(target_values):
        raise ValueError(
            f"FC-LSTM 输入窗口长度为 {len(legacy_fc_inputs)}，但 Figure 8 目标长度为 {len(target_values)}，无法对齐。"
        )
    fc_lstm_values = predict_fc_lstm_legacy(fc_lstm, legacy_fc_inputs, dataset.target_max, device)

    output_frame = pd.DataFrame(
        {
            "time_step": np.arange(len(target_values)),
            "target": target_values,
            "fm_gru": fm_gru_values,
            "fc_lstm": fc_lstm_values,
        }
    )
    output_frame.to_csv(table_dir / "figure8_trained_comparison.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(output_frame["time_step"], output_frame["fc_lstm"], color="#58C3AE", linewidth=1.8, label="FC-LSTM")
    ax.plot(output_frame["time_step"], output_frame["target"], color="#626B70", linewidth=1.8, label="Target")
    ax.plot(output_frame["time_step"], output_frame["fm_gru"], color="#E79284", linewidth=1.8, label="FM-GRU")
    ax.set_xlabel("time step")
    ax.set_ylabel("DO mg/L")
    ax.legend(loc="upper right")
    ax.set_title("Figure 8. Trained comparison of FC-LSTM, Target and FM-GRU")
    fig.tight_layout()
    fig.savefig(figure_dir / "figure8_trained_comparison.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source": "trained_from_scratch",
        "data_path": args.data_path,
        "epochs": args.epochs,
        "fm_batch_size": args.fm_batch_size,
        "fc_batch_size": args.fc_batch_size,
        "lr": args.lr,
        "hidden_size": args.hidden_size,
        "fm_num_layers": args.fm_num_layers,
        "fc_num_layers": args.fc_num_layers,
        "dropout": args.dropout,
        "teacher_prob": args.teacher_prob,
        "fm_k": args.fm_k,
        "plot_sample_indices": plot_sample_indices,
        "fc_train_data_path": str(fc_training_path),
        "fc_window_path": str(fc_window_path) if fc_window_path is not None else None,
        "fc_train_size": fc_train_size,
        "fc_target_multiplier": dataset.target_max,
        "static_prediction_extraction_used": False,
        "device": str(device),
        "elapsed_seconds": time.time() - start_time,
    }
    (table_dir / "figure8_trained_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("已从头训练生成 Figure 8:")
    print(figure_dir / "figure8_trained_comparison.png")
    print(table_dir / "figure8_trained_comparison.csv")


if __name__ == "__main__":
    main()
