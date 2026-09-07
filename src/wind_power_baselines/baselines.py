from __future__ import annotations

import pandas as pd

from wind_power_baselines.power_curve import estimate_power_from_curve
from wind_power_baselines.preprocessing import clean_monthly_truth, require_columns


def _target_months(monthly_truth: pd.DataFrame) -> list[str]:
    months = monthly_truth["month"].tolist()
    if len(months) < 2:
        raise ValueError("至少需要两个目标月份才能构造样例基线")
    return months[-2:]


def _truth_lookup(monthly_truth: pd.DataFrame) -> dict[str, float]:
    return {str(row.month): float(row.energy_true) for row in monthly_truth.itertuples(index=False)}


def _forecast_month_energy(forecast: pd.DataFrame, power_curve: pd.DataFrame) -> pd.DataFrame:
    require_columns(forecast, ["month", "wind_speed", "season"], "预处理后气象预报数据")
    data = forecast.copy()
    data["power_kw"] = estimate_power_from_curve(data["wind_speed"], data["season"], power_curve)
    monthly = data.groupby("month", as_index=False).agg(mean_wind_speed=("wind_speed", "mean"), mean_power_kw=("power_kw", "mean"))
    monthly["curve_energy"] = monthly["mean_power_kw"] * 24.0 * 30.0
    return monthly


def _prediction_frame(method: str, values: dict[str, float], truth: dict[str, float]) -> pd.DataFrame:
    rows = [{"method": method, "month": month, "energy_pred": prediction, "energy_true": truth[month]} for month, prediction in values.items()]
    return pd.DataFrame(rows)


def baseline_1_historical_same_month_mean(monthly_truth: pd.DataFrame) -> pd.DataFrame:
    truth = clean_monthly_truth(monthly_truth)
    target_months = _target_months(truth)
    truth_map = _truth_lookup(truth)
    predictions: dict[str, float] = {}
    for month in target_months:
        month_number = month[-2:]
        history = truth[(truth["month"] < month) & truth["month"].str.endswith(month_number)]
        if history.empty:
            history = truth[truth["month"] < month]
        if history.empty:
            raise ValueError(f"Baseline 1 无可用历史样本，目标月份={month}")
        predictions[month] = float(history["energy_true"].mean())
    return _prediction_frame("Baseline 1", predictions, truth_map)


def baseline_2_era5_p50(monthly_truth: pd.DataFrame, forecast: pd.DataFrame, power_curve: pd.DataFrame) -> pd.DataFrame:
    truth = clean_monthly_truth(monthly_truth)
    target_months = _target_months(truth)
    truth_map = _truth_lookup(truth)
    curve_energy = _forecast_month_energy(forecast, power_curve)
    median_curve = float(curve_energy["curve_energy"].median())
    predictions = {month: median_curve for month in target_months}
    return _prediction_frame("Baseline 2", predictions, truth_map)


def baseline_3_ec45_lls_power_curve(monthly_truth: pd.DataFrame, forecast: pd.DataFrame, power_curve: pd.DataFrame) -> pd.DataFrame:
    truth = clean_monthly_truth(monthly_truth)
    target_months = _target_months(truth)
    truth_map = _truth_lookup(truth)
    monthly = _forecast_month_energy(forecast, power_curve)
    predictions: dict[str, float] = {}
    for month in target_months:
        current = monthly[monthly["month"].eq(month)]
        if current.empty:
            raise ValueError(f"Baseline 3 缺少目标月份预报，目标月份={month}")
        predictions[month] = float(current["curve_energy"].iloc[0])
    return _prediction_frame("Baseline 3", predictions, truth_map)


def baseline_4_similar_day(monthly_truth: pd.DataFrame, forecast: pd.DataFrame, power_curve: pd.DataFrame) -> pd.DataFrame:
    truth = clean_monthly_truth(monthly_truth)
    target_months = _target_months(truth)
    truth_map = _truth_lookup(truth)
    monthly = _forecast_month_energy(forecast, power_curve)
    predictions: dict[str, float] = {}
    for month in target_months:
        current = monthly[monthly["month"].eq(month)]
        history = truth[truth["month"] < month]
        if current.empty or history.empty:
            raise ValueError(f"Baseline 4 缺少相似日样例输入，目标月份={month}")
        current_energy = float(current["curve_energy"].iloc[0])
        nearest_index = (history["energy_true"] - current_energy).abs().idxmin()
        predictions[month] = float(history.loc[nearest_index, "energy_true"])
    return _prediction_frame("Baseline 4", predictions, truth_map)


def ak_d_monthly_correction(monthly_truth: pd.DataFrame, forecast: pd.DataFrame, power_curve: pd.DataFrame, correction: pd.DataFrame) -> pd.DataFrame:
    require_columns(correction, ["month", "correction_factor"], "AK-D分月订正样例数据")
    truth = clean_monthly_truth(monthly_truth)
    target_months = _target_months(truth)
    truth_map = _truth_lookup(truth)
    monthly = _forecast_month_energy(forecast, power_curve)
    correction_data = correction.copy()
    correction_data["month"] = pd.to_datetime(correction_data["month"].astype(str) + "-01", errors="raise").dt.to_period("M").astype(str)
    correction_data["correction_factor"] = pd.to_numeric(correction_data["correction_factor"], errors="raise")
    predictions: dict[str, float] = {}
    for month in target_months:
        current = monthly[monthly["month"].eq(month)]
        factor = correction_data[correction_data["month"].eq(month)]
        if current.empty or factor.empty:
            raise ValueError(f"AK-D 缺少目标月份预报或分月订正系数，目标月份={month}")
        predictions[month] = float(current["curve_energy"].iloc[0]) * float(factor["correction_factor"].iloc[0])
    return _prediction_frame("AK-D", predictions, truth_map)


def build_all_baselines(monthly_truth: pd.DataFrame, forecast: pd.DataFrame, power_curve: pd.DataFrame, correction: pd.DataFrame) -> pd.DataFrame:
    frames = [
        baseline_1_historical_same_month_mean(monthly_truth),
        baseline_2_era5_p50(monthly_truth, forecast, power_curve),
        baseline_3_ec45_lls_power_curve(monthly_truth, forecast, power_curve),
        baseline_4_similar_day(monthly_truth, forecast, power_curve),
        ak_d_monthly_correction(monthly_truth, forecast, power_curve, correction),
    ]
    return pd.concat(frames, ignore_index=True)
