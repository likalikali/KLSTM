"""论文图表重算脚本共用的实验工具。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import random
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm.auto import tqdm

from repro.src.baseline_models import FeedForwardMultiStepModel, Seq2SeqRNNBaseline
from repro.src.data import DatasetConfig, FMGRUDataset, build_train_test_split
from repro.src.fm_gru_model import FMGRUModel
from repro.src.metrics import average_metric_dicts, compute_metrics


@dataclass
class PaperExperimentConfig:
    data_path: str = "dataset2.csv"
    target_column: str = "Dissolved oxygen"
    feature_columns: tuple[str, ...] | None = None
    encode_step: int = 24
    forecast_step: int = 12
    epochs: int = 10
    batch_size: int = 4
    lr: float = 0.001
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    teacher_prob: float = 0.8
    fm_k: int = 84
    train_ratio: float = 0.8
    train_index_path: str | None = "trainindex.csv"
    test_index_path: str | None = "testindex.csv"
    seed: int = 42

    def dataset_config(self) -> DatasetConfig:
        feature_columns = self.feature_columns or default_feature_columns(self.target_column)
        return DatasetConfig(
            data_path=self.data_path,
            encode_step=self.encode_step,
            forecast_step=self.forecast_step,
            feature_columns=feature_columns,
            target_column=self.target_column,
        )


@dataclass
class WindowArrays:
    flat_inputs: np.ndarray
    history_targets_original: np.ndarray
    future_targets_scaled: np.ndarray
    future_targets_original: np.ndarray


def default_feature_columns(target_column: str) -> tuple[str, ...]:
    canonical = ["Temperature", "Turbidity", "Conductivity", "PH", "Dissolved oxygen"]
    return tuple(column for column in canonical if column != target_column)


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


def require_optional_dependency(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(f"缺少依赖 `{module_name}`。请先在 .venv 中执行 `{install_hint}`。")


def read_index_file(path_like: str) -> list[int]:
    frame = pd.read_csv(path_like)
    return frame.iloc[:, 0].astype(int).tolist()


def resolve_split_indices(dataset: FMGRUDataset, config: PaperExperimentConfig, seed: int) -> tuple[list[int], list[int], dict]:
    if config.train_index_path:
        train_indices = read_index_file(config.train_index_path)
        if config.test_index_path:
            test_indices = read_index_file(config.test_index_path)
        else:
            train_index_set = set(train_indices)
            test_indices = [index for index in range(len(dataset)) if index not in train_index_set]
        return train_indices, test_indices, {
            "split_mode": "fixed_indices",
            "train_index_path": config.train_index_path,
            "test_index_path": config.test_index_path,
            "train_size": len(train_indices),
            "test_size": len(test_indices),
        }

    trainset, testset = build_train_test_split(dataset, config.train_ratio, seed)
    return list(trainset.indices), list(testset.indices), {
        "split_mode": "random_split",
        "train_ratio": config.train_ratio,
        "train_size": len(trainset),
        "test_size": len(testset),
    }


def build_window_arrays(dataset: FMGRUDataset, indices: list[int]) -> WindowArrays:
    flat_inputs = []
    history_targets_original = []
    future_targets_scaled = []
    future_targets_original = []

    for sample_index in indices:
        hisx, hisz, futx, z = dataset[sample_index]
        future = z[-dataset.forecast_step :, :].numpy()
        flat_inputs.append(
            np.concatenate((hisx.numpy().reshape(-1), hisz.numpy().reshape(-1), futx.numpy().reshape(-1))).astype(
                np.float32
            )
        )
        history_targets_original.append(dataset.inverse_target_transform(hisz.numpy()).reshape(-1).astype(np.float32))
        future_targets_scaled.append(future.reshape(-1).astype(np.float32))
        future_targets_original.append(dataset.inverse_target_transform(future).reshape(-1).astype(np.float32))

    return WindowArrays(
        flat_inputs=np.asarray(flat_inputs, dtype=np.float32),
        history_targets_original=np.asarray(history_targets_original, dtype=np.float32),
        future_targets_scaled=np.asarray(future_targets_scaled, dtype=np.float32),
        future_targets_original=np.asarray(future_targets_original, dtype=np.float32),
    )


def build_loader_from_indices(
    dataset: FMGRUDataset,
    indices: list[int],
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _evaluate_forecasts(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return compute_metrics(predictions.reshape(-1), targets.reshape(-1))


def _repeat_iterator(repeats: int, description: str):
    return tqdm(range(repeats), desc=description, dynamic_ncols=True)


def train_fm_gru_variant(
    dataset: FMGRUDataset,
    train_indices: list[int],
    config: PaperExperimentConfig,
    device: torch.device,
    use_fm: bool,
    progress_label: str,
) -> FMGRUModel:
    loader = build_loader_from_indices(dataset, train_indices, config.batch_size, shuffle=False, drop_last=True)
    model = FMGRUModel(
        target_size=1,
        feature_size=len(config.dataset_config().feature_columns),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        forecast_step=config.forecast_step,
        encode_step=config.encode_step,
        teacher_prob=config.teacher_prob,
        fm_k=config.fm_k,
        use_fm=use_fm,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_func = torch.nn.MSELoss()

    total_steps = max(len(loader), 1) * config.epochs
    progress = tqdm(total=total_steps, desc=progress_label, leave=False, dynamic_ncols=True)
    for epoch_index in range(config.epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for hisx, hisz, futx, z in loader:
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
            running_loss += float(loss.detach().item())
            batch_count += 1
            progress.update(1)
        progress.set_postfix(
            epoch=f"{epoch_index + 1}/{config.epochs}",
            loss=f"{running_loss / max(batch_count, 1):.4f}",
        )
    progress.close()

    return model


def evaluate_fm_gru_variant(
    model: FMGRUModel,
    dataset: FMGRUDataset,
    test_indices: list[int],
    batch_size: int,
    forecast_step: int,
    device: torch.device,
) -> dict[str, float]:
    loader = build_loader_from_indices(dataset, test_indices, batch_size, shuffle=False, drop_last=False)
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for hisx, hisz, futx, z in loader:
            hisx = hisx.to(device)
            hisz = hisz.to(device)
            futx = futx.to(device)
            z = z.to(device)

            _, forecast, _ = model(hisx, hisz, futx, z)
            future = z[:, -forecast_step :, :]
            preds.append(dataset.inverse_target_transform(forecast.detach().cpu().numpy()).reshape(len(z), -1))
            trues.append(dataset.inverse_target_transform(future.detach().cpu().numpy()).reshape(len(z), -1))

    return _evaluate_forecasts(np.vstack(preds), np.vstack(trues))


def run_fm_gru_metrics(
    config: PaperExperimentConfig,
    device: torch.device,
    repeats: int = 1,
    use_fm: bool = True,
) -> tuple[dict[str, float], dict]:
    metric_rows = []
    split_info = None
    model_label = "FM-GRU" if use_fm else "Baseline"
    repeat_progress = _repeat_iterator(repeats, f"{model_label} repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        set_seed(seed)
        dataset = FMGRUDataset(config.dataset_config())
        train_indices, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        model = train_fm_gru_variant(
            dataset,
            train_indices,
            config,
            device,
            use_fm=use_fm,
            progress_label=f"{model_label} train {repeat_index + 1}/{repeats}",
        )
        metrics = evaluate_fm_gru_variant(model, dataset, test_indices, config.batch_size, config.forecast_step, device)
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_seq2seq_baseline_metrics(
    config: PaperExperimentConfig,
    device: torch.device,
    cell_type: str,
    repeats: int = 1,
) -> tuple[dict[str, float], dict]:
    metric_rows = []
    split_info = None
    model_label = f"FC-{cell_type.upper()}"
    repeat_progress = _repeat_iterator(repeats, f"{model_label} repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        set_seed(seed)
        dataset = FMGRUDataset(config.dataset_config())
        train_indices, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        train_loader = build_loader_from_indices(dataset, train_indices, config.batch_size, shuffle=True, drop_last=True)
        test_loader = build_loader_from_indices(dataset, test_indices, config.batch_size, shuffle=False, drop_last=False)

        model = Seq2SeqRNNBaseline(
            feature_size=len(config.dataset_config().feature_columns),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            forecast_step=config.forecast_step,
            cell_type=cell_type,
            dropout=config.dropout,
            teacher_prob=config.teacher_prob,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_func = torch.nn.MSELoss()

        total_steps = max(len(train_loader), 1) * config.epochs
        progress = tqdm(
            total=total_steps,
            desc=f"{model_label} train {repeat_index + 1}/{repeats}",
            leave=False,
            dynamic_ncols=True,
        )
        for epoch_index in range(config.epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for hisx, hisz, futx, z in train_loader:
                hisx = hisx.to(device)
                hisz = hisz.to(device)
                futx = futx.to(device)
                z = z.to(device)
                future_target = z[:, -config.forecast_step :, :]
                prediction = model(hisx, hisz, futx, z)
                loss = loss_func(prediction, future_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().item())
                batch_count += 1
                progress.update(1)
            progress.set_postfix(
                epoch=f"{epoch_index + 1}/{config.epochs}",
                loss=f"{running_loss / max(batch_count, 1):.4f}",
            )
        progress.close()

        preds = []
        trues = []
        model.eval()
        with torch.no_grad():
            for hisx, hisz, futx, z in test_loader:
                hisx = hisx.to(device)
                hisz = hisz.to(device)
                futx = futx.to(device)
                z = z.to(device)
                future_target = z[:, -config.forecast_step :, :]
                prediction = model(hisx, hisz, futx, z)
                preds.append(dataset.inverse_target_transform(prediction.detach().cpu().numpy()).reshape(len(z), -1))
                trues.append(
                    dataset.inverse_target_transform(future_target.detach().cpu().numpy()).reshape(len(z), -1)
                )

        metrics = _evaluate_forecasts(np.vstack(preds), np.vstack(trues))
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_ffnn_metrics(
    config: PaperExperimentConfig,
    device: torch.device,
    repeats: int = 1,
) -> tuple[dict[str, float], dict]:
    metric_rows = []
    split_info = None
    repeat_progress = _repeat_iterator(repeats, "FFNN repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        set_seed(seed)
        dataset = FMGRUDataset(config.dataset_config())
        train_indices, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        train_arrays = build_window_arrays(dataset, train_indices)
        test_arrays = build_window_arrays(dataset, test_indices)

        model = FeedForwardMultiStepModel(
            input_size=train_arrays.flat_inputs.shape[1],
            output_size=train_arrays.future_targets_original.shape[1],
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_func = torch.nn.MSELoss()
        train_dataset = TensorDataset(
            torch.from_numpy(train_arrays.flat_inputs),
            torch.from_numpy(train_arrays.future_targets_original),
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

        total_steps = max(len(train_loader), 1) * config.epochs
        progress = tqdm(
            total=total_steps,
            desc=f"FFNN train {repeat_index + 1}/{repeats}",
            leave=False,
            dynamic_ncols=True,
        )
        for epoch_index in range(config.epochs):
            model.train()
            running_loss = 0.0
            batch_count = 0
            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)
                prediction = model(features)
                loss = loss_func(prediction, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().item())
                batch_count += 1
                progress.update(1)
            progress.set_postfix(
                epoch=f"{epoch_index + 1}/{config.epochs}",
                loss=f"{running_loss / max(batch_count, 1):.4f}",
            )
        progress.close()

        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(test_arrays.flat_inputs).to(device)).cpu().numpy()
        metrics = _evaluate_forecasts(prediction, test_arrays.future_targets_original)
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_lr_metrics(config: PaperExperimentConfig, repeats: int = 1) -> tuple[dict[str, float], dict]:
    metric_rows = []
    split_info = None
    repeat_progress = _repeat_iterator(repeats, "LR repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        dataset = FMGRUDataset(config.dataset_config())
        train_indices, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        train_arrays = build_window_arrays(dataset, train_indices)
        test_arrays = build_window_arrays(dataset, test_indices)
        model = LinearRegression()
        model.fit(train_arrays.flat_inputs, train_arrays.future_targets_original)
        prediction = model.predict(test_arrays.flat_inputs)
        metrics = _evaluate_forecasts(prediction, test_arrays.future_targets_original)
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_xgboost_metrics(
    config: PaperExperimentConfig,
    repeats: int = 1,
) -> tuple[dict[str, float], dict]:
    require_optional_dependency("xgboost", "pip install xgboost")
    from xgboost import XGBRegressor

    metric_rows = []
    split_info = None
    repeat_progress = _repeat_iterator(repeats, "XGBoost repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        dataset = FMGRUDataset(config.dataset_config())
        train_indices, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        train_arrays = build_window_arrays(dataset, train_indices)
        test_arrays = build_window_arrays(dataset, test_indices)

        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
        model = MultiOutputRegressor(estimator)
        model.fit(train_arrays.flat_inputs, train_arrays.future_targets_original)
        prediction = model.predict(test_arrays.flat_inputs)
        metrics = _evaluate_forecasts(prediction, test_arrays.future_targets_original)
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_ha_metrics(config: PaperExperimentConfig, repeats: int = 1) -> tuple[dict[str, float], dict]:
    metric_rows = []
    split_info = None
    repeat_progress = _repeat_iterator(repeats, "HA repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        dataset = FMGRUDataset(config.dataset_config())
        _, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        test_arrays = build_window_arrays(dataset, test_indices)
        predictions = np.repeat(test_arrays.history_targets_original.mean(axis=1, keepdims=True), config.forecast_step, axis=1)
        metrics = _evaluate_forecasts(predictions, test_arrays.future_targets_original)
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_arima_metrics(config: PaperExperimentConfig, repeats: int = 1) -> tuple[dict[str, float], dict]:
    require_optional_dependency("statsmodels", "pip install statsmodels")
    from statsmodels.tsa.arima.model import ARIMA

    metric_rows = []
    split_info = None
    repeat_progress = _repeat_iterator(repeats, "ARIMA repeats")
    for repeat_index in repeat_progress:
        seed = config.seed + repeat_index
        dataset = FMGRUDataset(config.dataset_config())
        _, test_indices, split_info = resolve_split_indices(dataset, config, seed)
        test_arrays = build_window_arrays(dataset, test_indices)
        predictions = []
        sample_progress = tqdm(
            zip(test_arrays.history_targets_original, test_arrays.future_targets_original),
            total=len(test_arrays.history_targets_original),
            desc=f"ARIMA forecast {repeat_index + 1}/{repeats}",
            leave=False,
            dynamic_ncols=True,
        )
        for history, future in sample_progress:
            model = ARIMA(history.astype(np.float64), order=(1, 2, 1))
            fit = model.fit()
            forecast = np.asarray(fit.forecast(steps=len(future)), dtype=np.float32)
            predictions.append(forecast)
        sample_progress.close()
        metrics = _evaluate_forecasts(np.asarray(predictions, dtype=np.float32), test_arrays.future_targets_original)
        metric_rows.append(metrics)
        repeat_progress.set_postfix(rmse=f"{metrics['RMSE']:.4f}")
    repeat_progress.close()
    return average_metric_dicts(metric_rows), split_info or {}


def run_model_suite(
    config: PaperExperimentConfig,
    device: torch.device,
    models: list[str],
    repeats: int,
) -> tuple[pd.DataFrame, dict]:
    runners: dict[str, Callable[[], tuple[dict[str, float], dict]]] = {
        "HA": lambda: run_ha_metrics(config, repeats=repeats),
        "ARIMA": lambda: run_arima_metrics(config, repeats=repeats),
        "LR": lambda: run_lr_metrics(config, repeats=repeats),
        "XGBoost": lambda: run_xgboost_metrics(config, repeats=repeats),
        "FFNN": lambda: run_ffnn_metrics(config, device=device, repeats=repeats),
        "FC-LSTM": lambda: run_seq2seq_baseline_metrics(config, device=device, cell_type="lstm", repeats=repeats),
        "FC-GRU": lambda: run_seq2seq_baseline_metrics(config, device=device, cell_type="gru", repeats=repeats),
        "FM-GRU": lambda: run_fm_gru_metrics(config, device=device, repeats=repeats, use_fm=True),
    }

    records = []
    split_info = {}
    model_progress = tqdm(models, desc="Models", dynamic_ncols=True)
    for model_name in model_progress:
        metrics, split_info = runners[model_name]()
        row = {"Model": model_name}
        row.update({metric_name: metrics[metric_name] for metric_name in ["MAE", "MSE", "RMSE", "NRMSE"] if metric_name in metrics})
        records.append(row)
        if "RMSE" in metrics:
            model_progress.set_postfix(model=model_name, rmse=f"{metrics['RMSE']:.4f}")
    model_progress.close()
    return pd.DataFrame(records), split_info
