"""数据加载与缩放逻辑，统一服务于 FM-GRU 训练和图表脚本。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_COLUMNS = ["Temperature", "Turbidity", "Conductivity", "PH", "Dissolved oxygen"]


def resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_csv_with_fallback(path_like: str | Path, encodings: Iterable[str] | None = None) -> pd.DataFrame:
    path = resolve_project_path(path_like)
    encodings = tuple(encodings or ("utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"))
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path)


def canonicalize_water_quality_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.shape[1] < 5:
        raise ValueError("输入数据至少需要包含 5 列水质字段。")
    data = frame.iloc[:, :5].copy()
    data.columns = CANONICAL_COLUMNS
    for column in CANONICAL_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # 论文原始训练代码会保留全空行，再通过 fillna/ffill 参与后续处理。
    return data.reset_index(drop=True)


def load_water_quality_dataframe(path_like: str | Path) -> pd.DataFrame:
    return canonicalize_water_quality_dataframe(read_csv_with_fallback(path_like))


def minmax_scale_array(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=np.float32).copy()
    mins = np.nanmin(values, axis=0)
    maxs = np.nanmax(values, axis=0)
    scale = maxs - mins
    scale[scale == 0] = 1.0
    values = (values - mins) / scale
    return values.astype(np.float32), mins.astype(np.float32), maxs.astype(np.float32)


def inverse_minmax(data: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32).copy()
    mins = np.asarray(mins, dtype=np.float32)
    maxs = np.asarray(maxs, dtype=np.float32)
    scale = maxs - mins
    scale[scale == 0] = 1.0
    return values * scale + mins


def _fill_missing(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "zero":
        return frame.fillna(0)
    if method == "ffill":
        return frame.ffill()
    if method == "bfill":
        return frame.bfill()
    raise ValueError(f"不支持的缺失值处理方式: {method}")


@dataclass
class DatasetConfig:
    data_path: str = "KRMData/oxy/原始数据平滑训练和验证集不平滑测试集.csv"
    encode_step: int = 24
    forecast_step: int = 12
    feature_columns: Sequence[str] = ("Temperature", "Turbidity", "Conductivity", "PH")
    target_column: str = "Dissolved oxygen"
    feature_fill_method: str = "zero"
    target_fill_method: str = "ffill"
    double_scale_features: bool = True


class FMGRUDataset(Dataset):
    """尽量保持与原论文代码一致的数据切片逻辑。"""

    def __init__(self, config: DatasetConfig):
        super().__init__()
        self.config = config
        self.encode_step = config.encode_step
        self.forecast_step = config.forecast_step

        raw = load_water_quality_dataframe(config.data_path)
        feature_frame = _fill_missing(raw[list(config.feature_columns)], config.feature_fill_method)
        target_frame = _fill_missing(raw[[config.target_column]], config.target_fill_method)

        features, self.feature_mins, self.feature_maxs = minmax_scale_array(feature_frame.to_numpy())
        if config.double_scale_features:
            features, _, _ = minmax_scale_array(features)

        target, target_mins, target_maxs = minmax_scale_array(target_frame.to_numpy())
        self.target_min = float(target_mins[0])
        self.target_max = float(target_maxs[0])
        self.scaler = StandardScaler()
        self.scaler.fit(target)
        target = self.scaler.transform(target).astype(np.float32)

        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.raw_dataframe = raw

    def __len__(self) -> int:
        return len(self.features) - self.forecast_step - self.encode_step - 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        index += 1

        his_start = index
        his_end = his_start + self.encode_step
        fut_start = his_end + 1
        fut_end = fut_start + self.forecast_step

        hisx = self.features[his_start:his_end]
        hisz = self.target[his_start - 1 : his_end - 1]
        futx = self.features[fut_start:fut_end]
        z = self.target[index : index + self.encode_step + self.forecast_step]

        return (
            torch.from_numpy(hisx),
            torch.from_numpy(hisz),
            torch.from_numpy(futx),
            torch.from_numpy(z),
        )

    def inverse_target_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        original_shape = array.shape
        restored = self.scaler.inverse_transform(array.reshape(-1, 1)).reshape(original_shape)
        restored = inverse_minmax(
            restored.reshape(-1, 1),
            np.asarray([self.target_min], dtype=np.float32),
            np.asarray([self.target_max], dtype=np.float32),
        )
        return restored.reshape(original_shape)


def build_train_test_split(dataset: Dataset, train_ratio: float, seed: int):
    train_size = int(len(dataset) * train_ratio)
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)
