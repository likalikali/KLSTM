"""训练与出图共用的误差指标。"""

from __future__ import annotations

import numpy as np


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    true = np.asarray(true, dtype=np.float32).reshape(-1)

    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))

    t1 = np.sum((pred - true) ** 2) / np.size(true)
    t2 = np.sum(np.abs(true)) / np.size(true)
    nrmse = float(np.sqrt(t1) / t2) if t2 != 0 else float("nan")

    ss_tot = np.sum((true - np.mean(pred)) ** 2)
    ss_res = np.sum((true - pred) ** 2)
    r = float(1 - ss_res / ss_tot) if ss_tot != 0 else float("nan")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "R": r}


def average_metric_dicts(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    return {key: float(np.mean([item[key] for item in metrics_list])) for key in keys}
