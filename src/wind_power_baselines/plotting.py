from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from wind_power_baselines.preprocessing import require_columns


def plot_power_curve(power_curve: pd.DataFrame, output_path: Path) -> Path:
    require_columns(power_curve, ["turbine_id", "season", "wind_speed_bin", "power_kw"], "power curve data")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (turbine_id, season), group in power_curve.groupby(["turbine_id", "season"]):
        ax.plot(group["wind_speed_bin"], group["power_kw"], marker="o", label=f"{turbine_id}-{season}")
    ax.set_xlabel("Wind speed bin (m/s)")
    ax.set_ylabel("Median power (kW)")
    ax.set_title("Empirical power curve")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_metric_bars(metrics: pd.DataFrame, output_path: Path) -> Path:
    require_columns(metrics, ["method", "mape_percent", "bias_percent"], "metric data")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics.plot.bar(x="method", y="mape_percent", ax=axes[0], legend=False, color="#4C78A8")
    metrics.plot.bar(x="method", y="bias_percent", ax=axes[1], legend=False, color="#F58518")
    axes[0].set_ylabel("MAPE (%)")
    axes[1].set_ylabel("Bias (%)")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
