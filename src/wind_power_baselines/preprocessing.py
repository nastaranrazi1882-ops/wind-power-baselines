from __future__ import annotations

from collections.abc import Sequence
import pandas as pd


def require_columns(data: pd.DataFrame, required_columns: Sequence[str], dataset_name: str) -> None:
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}; actual columns: {list(data.columns)}")


def season_from_month(month_values: pd.Series) -> pd.Series:
    month_numbers = pd.to_datetime(month_values.astype(str) + "-01", errors="raise").dt.month
    return month_numbers.map(lambda month: "winter" if month in (11, 12, 1, 2, 3) else "warm")


def clean_scada(raw_scada: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw_scada, ["timestamp", "turbine_id", "wind_speed", "power_kw", "status"], "SCADA sample data")
    data = raw_scada.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["wind_speed"] = pd.to_numeric(data["wind_speed"], errors="coerce")
    data["power_kw"] = pd.to_numeric(data["power_kw"], errors="coerce")
    data["turbine_id"] = data["turbine_id"].astype(str)
    data["status"] = data["status"].astype(str)
    valid = data[data["timestamp"].notna() & data["wind_speed"].between(0.0, 35.0) & data["power_kw"].between(0.0, 5000.0) & data["status"].str.lower().eq("normal")].copy()
    if valid.empty:
        raise ValueError("SCADA sample data is empty after cleaning")
    valid = valid.drop_duplicates(subset=["timestamp", "turbine_id"]).sort_values(["turbine_id", "timestamp"])
    valid["month"] = valid["timestamp"].dt.to_period("M").astype(str)
    valid["season"] = season_from_month(valid["month"])
    return valid.reset_index(drop=True)


def prepare_forecast(raw_forecast: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw_forecast, ["issue_time", "target_time", "lead_day", "wind_speed"], "forecast sample data")
    data = raw_forecast.copy()
    data["issue_time"] = pd.to_datetime(data["issue_time"], errors="raise")
    data["target_time"] = pd.to_datetime(data["target_time"], errors="raise")
    data["lead_day"] = pd.to_numeric(data["lead_day"], errors="raise").astype(int)
    data["wind_speed"] = pd.to_numeric(data["wind_speed"], errors="raise")
    invalid = data[(data["lead_day"] < 0) | ~data["wind_speed"].between(0.0, 35.0)]
    if not invalid.empty:
        raise ValueError(f"forecast sample data has invalid lead_day or wind_speed rows: {len(invalid)}")
    data["month"] = data["target_time"].dt.to_period("M").astype(str)
    data["issue_month"] = data["issue_time"].dt.to_period("M").astype(str)
    data["season"] = season_from_month(data["month"])
    return data.sort_values(["issue_time", "target_time"]).reset_index(drop=True)


def clean_monthly_truth(raw_truth: pd.DataFrame) -> pd.DataFrame:
    require_columns(raw_truth, ["month", "energy_true"], "monthly truth sample data")
    data = raw_truth.copy()
    data["month"] = pd.to_datetime(data["month"].astype(str) + "-01", errors="raise").dt.to_period("M").astype(str)
    data["energy_true"] = pd.to_numeric(data["energy_true"], errors="raise")
    invalid = data[data["energy_true"] <= 0.0]
    if not invalid.empty:
        raise ValueError(f"monthly truth must be positive; invalid rows: {len(invalid)}")
    return data.drop_duplicates(subset=["month"]).sort_values("month").reset_index(drop=True)
