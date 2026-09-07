from __future__ import annotations

import numpy as np
import pandas as pd

from wind_power_baselines.preprocessing import require_columns


def assign_wind_speed_bin(wind_speed: pd.Series, bin_width: float) -> pd.Series:
    if bin_width <= 0.0:
        raise ValueError(f"风速分箱宽度必须大于0，收到={bin_width}")
    return (np.floor(wind_speed.astype(float) / bin_width) * bin_width).round(3)


def build_power_curve(cleaned_scada: pd.DataFrame, bin_width: float, min_samples: int) -> pd.DataFrame:
    require_columns(cleaned_scada, ["turbine_id", "season", "wind_speed", "power_kw"], "清洗后SCADA数据")
    if min_samples <= 0:
        raise ValueError(f"每个风速箱最小样本数必须大于0，收到={min_samples}")
    data = cleaned_scada.copy()
    data["wind_speed_bin"] = assign_wind_speed_bin(data["wind_speed"], bin_width)
    grouped = (
        data.groupby(["turbine_id", "season", "wind_speed_bin"], as_index=False)
        .agg(power_kw=("power_kw", "median"), sample_count=("power_kw", "size"))
        .sort_values(["turbine_id", "season", "wind_speed_bin"])
    )
    curve = grouped[grouped["sample_count"] >= min_samples].copy()
    if curve.empty:
        raise ValueError(f"功率曲线为空，请降低min_samples或检查SCADA覆盖，min_samples={min_samples}")
    return curve.reset_index(drop=True)


def estimate_power_from_curve(wind_speed: pd.Series, season: pd.Series, power_curve: pd.DataFrame) -> pd.Series:
    require_columns(power_curve, ["season", "wind_speed_bin", "power_kw"], "功率曲线数据")
    global_curve = power_curve.groupby(["season", "wind_speed_bin"], as_index=False)["power_kw"].mean()
    rows = pd.DataFrame({"wind_speed": wind_speed.astype(float), "season": season.astype(str)})
    estimates: list[float] = []
    for row in rows.itertuples(index=False):
        candidates = global_curve[global_curve["season"].eq(row.season)]
        if candidates.empty:
            candidates = global_curve
        nearest_index = (candidates["wind_speed_bin"] - float(row.wind_speed)).abs().idxmin()
        estimates.append(float(candidates.loc[nearest_index, "power_kw"]))
    return pd.Series(estimates, index=wind_speed.index, dtype="float64")
