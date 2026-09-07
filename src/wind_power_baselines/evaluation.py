from __future__ import annotations

import pandas as pd

from wind_power_baselines.preprocessing import require_columns


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    require_columns(predictions, ["method", "month", "energy_pred", "energy_true"], "预测结果")
    data = predictions.copy()
    data["energy_pred"] = pd.to_numeric(data["energy_pred"], errors="raise")
    data["energy_true"] = pd.to_numeric(data["energy_true"], errors="raise")
    if data["energy_true"].le(0.0).any():
        raise ValueError("预测结果中的真值必须全部为正数")
    data["absolute_percent_error"] = (data["energy_pred"] - data["energy_true"]).abs() / data["energy_true"] * 100.0
    rows: list[dict[str, float | str | int]] = []
    for method, group in data.groupby("method", sort=False):
        total_true = float(group["energy_true"].sum())
        total_pred = float(group["energy_pred"].sum())
        pearson_r = float("nan")
        if len(group) > 1 and float(group["energy_pred"].std()) > 0.0 and float(group["energy_true"].std()) > 0.0:
            pearson_r = round(float(group["energy_pred"].corr(group["energy_true"])), 4)
        rows.append(
            {
                "method": str(method),
                "month_count": int(len(group)),
                "mape_percent": round(float(group["absolute_percent_error"].mean()), 4),
                "bias_percent": round((total_pred - total_true) / total_true * 100.0, 4),
                "pearson_r": pearson_r,
            }
        )
    return pd.DataFrame(rows)


def normalize_public_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    require_columns(predictions, ["method", "month", "energy_pred", "energy_true"], "预测结果")
    data = predictions.copy()
    denominator = float(pd.to_numeric(data["energy_true"], errors="raise").mean())
    if denominator <= 0.0:
        raise ValueError("归一化分母必须大于0")
    public_data = data[["method", "month"]].copy()
    public_data["energy_true_index"] = pd.to_numeric(data["energy_true"], errors="raise") / denominator
    public_data["energy_pred_index"] = pd.to_numeric(data["energy_pred"], errors="raise") / denominator
    public_data["error_percent"] = (pd.to_numeric(data["energy_pred"], errors="raise") - pd.to_numeric(data["energy_true"], errors="raise")) / pd.to_numeric(data["energy_true"], errors="raise") * 100.0
    return public_data.round({"energy_true_index": 4, "energy_pred_index": 4, "error_percent": 4})
