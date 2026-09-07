from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["Baseline 1", "Baseline 2", "Baseline 3", "Baseline 4", "AK-D"]


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"找不到{label}: path={path}")
    if not path.is_file():
        raise IsADirectoryError(f"{label}不是文件: path={path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"找不到{label}: path={path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label}不是目录: path={path}")
    return path


def read_csv_checked(path: Path, columns: list[str], label: str) -> pd.DataFrame:
    checked_path = require_file(path, label)
    data = pd.read_csv(checked_path)
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label}缺少字段: path={checked_path}, missing={missing}, columns={list(data.columns)}")
    return data


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)
    return output_path


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 9


def render_wind_distribution(source_root: Path, output_root: Path) -> Path:
    wind_path = source_root / "baseline3" / "output" / "baseline3_ec45_middle45_timingfix" / "baseline_output" / "_cache_obs_wind_6h_per_turbine.csv"
    wind = read_csv_checked(wind_path, ["time", "turbine_1", "turbine_2", "turbine_3", "turbine_4", "turbine_5", "turbine_6"], "逐机观测风速缓存")
    wind["time"] = pd.to_datetime(wind["time"])
    wind_long = wind.melt(id_vars="time", var_name="turbine", value_name="wind_speed").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    sns.boxplot(data=wind_long, x="turbine", y="wind_speed", ax=axes[0], color="#78A8D8")
    sns.kdeplot(data=wind_long, x="wind_speed", hue="turbine", common_norm=False, ax=axes[1], linewidth=1.8)
    axes[0].set_title("Observed wind speed distribution by turbine")
    axes[0].set_xlabel("Turbine")
    axes[0].set_ylabel("Wind speed (m/s)")
    axes[1].set_title("Observed wind speed density")
    axes[1].set_xlabel("Wind speed (m/s)")
    return save_figure(fig, output_root / "power_curve" / "actual_wind_distribution_by_turbine.png")


def render_daily_wind_energy(source_root: Path, output_root: Path) -> Path:
    wind_path = source_root / "baseline3" / "output" / "baseline3_ec45_middle45_timingfix" / "baseline_output" / "_cache_obs_wind_6h.csv"
    energy_path = source_root / "baseline3" / "output" / "baseline3_ec45_middle45_timingfix" / "baseline_output" / "_cache_obs_energy_day.csv"
    wind = read_csv_checked(wind_path, ["time", "obs_wind_6h"], "场站观测风速缓存")
    energy = read_csv_checked(energy_path, ["time", "energy_true_day", "truth_available_ratio"], "日真值电量缓存")

    wind["time"] = pd.to_datetime(wind["time"])
    energy["date"] = pd.to_datetime(energy["time"]).dt.date
    wind_daily = wind.assign(date=wind["time"].dt.date).groupby("date", as_index=False)["obs_wind_6h"].mean()
    joined = wind_daily.merge(energy[["date", "energy_true_day", "truth_available_ratio"]], on="date", how="inner")
    joined = joined[joined["truth_available_ratio"] >= 0.99].copy()
    if joined.empty:
        raise ValueError(f"日风速与电量合并后为空: wind_path={wind_path}, energy_path={energy_path}")
    joined["energy_true_index"] = joined["energy_true_day"] / joined["energy_true_day"].mean()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.scatterplot(data=joined, x="obs_wind_6h", y="energy_true_index", alpha=0.55, s=32, color="#4C78A8", edgecolor=None, ax=ax)
    sns.regplot(data=joined, x="obs_wind_6h", y="energy_true_index", scatter=False, lowess=True, color="#D14B40", ax=ax)
    ax.set_title("Daily observed wind speed vs normalized energy")
    ax.set_xlabel("Daily mean observed wind speed (m/s)")
    ax.set_ylabel("Energy index (mean = 1)")
    return save_figure(fig, output_root / "power_curve" / "actual_daily_wind_energy_scatter.png")


def read_power_curve_candidates(source_root: Path) -> pd.DataFrame:
    curve_dir = source_root / "baseline4" / "data" / "功率曲线缓存" / "curve_candidates"
    checked_dir = require_directory(curve_dir, "功率曲线候选目录")
    curve_paths = sorted(glob.glob(str(checked_dir / "*.csv")))
    if not curve_paths:
        raise FileNotFoundError(f"功率曲线候选目录没有 CSV: path={checked_dir}")
    frames = [read_csv_checked(Path(path), ["wind_center", "median_smooth", "season_flag", "fan_id"], "功率曲线候选表") for path in curve_paths]
    curves = pd.concat(frames, ignore_index=True)
    curves = curves.dropna(subset=["wind_center", "median_smooth"]).copy()
    curves = curves[curves["median_smooth"] >= 0].copy()
    if curves.empty:
        raise ValueError(f"功率曲线候选表过滤后为空: path={checked_dir}")
    curves["turbine"] = "T" + curves["fan_id"].astype(int).astype(str).str.zfill(2)
    curves["season"] = curves["season_flag"].map({"cold": "Cold season", "warm": "Warm season"}).fillna(curves["season_flag"].astype(str))
    curves["power_index"] = curves.groupby("fan_id")["median_smooth"].transform(lambda values: values / values.max())
    return curves


def render_power_curve_all_turbines(curves: pd.DataFrame, output_root: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    sns.lineplot(data=curves, x="wind_center", y="power_index", hue="turbine", style="season", linewidth=2.0, ax=ax)
    ax.set_title("Actual fitted power curves by turbine")
    ax.set_xlabel("Wind speed bin center (m/s)")
    ax.set_ylabel("Power index (turbine max = 1)")
    ax.set_ylim(-0.03, 1.08)
    return save_figure(fig, output_root / "power_curve" / "actual_power_curve_all_turbines.png")


def render_power_curve_seasonal(curves: pd.DataFrame, output_root: Path) -> Path:
    seasonal = curves.groupby(["season", "wind_center"], as_index=False).agg(power_index=("power_index", "median"), turbine_count=("fan_id", "nunique"))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    sns.lineplot(data=seasonal, x="wind_center", y="power_index", hue="season", marker="o", linewidth=2.2, ax=axes[0])
    sns.barplot(data=seasonal, x="wind_center", y="turbine_count", hue="season", ax=axes[1])
    axes[0].set_title("Seasonal median fitted power curve")
    axes[0].set_xlabel("Wind speed bin center (m/s)")
    axes[0].set_ylabel("Power index")
    axes[1].set_title("Turbine coverage by wind-speed bin")
    axes[1].set_xlabel("Wind speed bin center (m/s)")
    axes[1].set_ylabel("Turbines with valid curve point")
    axes[1].tick_params(axis="x", rotation=90)
    return save_figure(fig, output_root / "power_curve" / "actual_power_curve_seasonal.png")


def render_metrics(metrics: pd.DataFrame, output_root: Path) -> Path:
    plot_data = metrics.copy()
    plot_data["method"] = pd.Categorical(plot_data["method"], categories=METHOD_ORDER, ordered=True)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.barplot(data=plot_data.sort_values("method"), x="method", y="mape_percent", ax=axes[0], palette="Blues_d", hue="method", legend=False)
    sns.barplot(data=plot_data.sort_values("method"), x="method", y="bias_percent", ax=axes[1], palette="RdBu_r", hue="method", legend=False)
    axes[0].set_title("MAPE comparison by baseline")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("MAPE (%)")
    axes[1].set_title("Bias comparison by baseline")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Bias (%)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        axis.axhline(0, color="black", linewidth=0.8)
    return save_figure(fig, output_root / "point_forecast" / "actual_baseline_mape_bias.png")


def render_monthly_index(monthly: pd.DataFrame, output_root: Path) -> Path:
    plot_data = monthly.copy()
    plot_data["method"] = pd.Categorical(plot_data["method"], categories=METHOD_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    sns.lineplot(data=plot_data, x="month", y="energy_pred_index", hue="method", marker="o", linewidth=1.8, ax=ax)
    truth_line = plot_data.drop_duplicates("month").sort_values("month")
    sns.lineplot(data=truth_line, x="month", y="energy_true_index", color="black", marker="o", linewidth=2.5, label="Truth index", ax=ax)
    ax.set_title("Monthly normalized prediction trajectories")
    ax.set_xlabel("Target month")
    ax.set_ylabel("Energy index")
    ax.tick_params(axis="x", rotation=45)
    return save_figure(fig, output_root / "point_forecast" / "actual_monthly_prediction_index.png")


def render_error_heatmap(monthly: pd.DataFrame, output_root: Path) -> Path:
    plot_data = monthly.copy()
    plot_data["method"] = pd.Categorical(plot_data["method"], categories=METHOD_ORDER, ordered=True)
    heatmap_data = plot_data.pivot_table(index="method", columns="month", values="error_percent", observed=False)
    fig, ax = plt.subplots(figsize=(16.0, 6.2))
    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 8},
        linewidths=0.4,
        cbar_kws={"label": "Error (%)"},
        ax=ax,
    )
    ax.set_title("Monthly relative error heatmap")
    ax.set_xlabel("Target month")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=60)
    ax.tick_params(axis="y", rotation=0)
    return save_figure(fig, output_root / "point_forecast" / "actual_monthly_error_heatmap.png")


def render_lead_day_error(source_root: Path, output_root: Path) -> Path:
    baseline3_path = source_root / "baseline3" / "output" / "baseline3_ec45_middle45_timingfix" / "baseline_output" / "metrics_by_ahead_days.csv"
    baseline4_path = source_root / "baseline4" / "output" / "baseline4_simday_ec45_wdir0_powercurve_timingfix" / "metrics_by_ahead_days.csv"
    baseline3 = read_csv_checked(baseline3_path, ["ahead_days", "mean_abs_rel_bias_pct", "pearson_r"], "Baseline 3提前期指标")
    baseline4 = read_csv_checked(baseline4_path, ["ahead_days", "mean_abs_rel_bias_pct", "pearson_r"], "Baseline 4提前期指标")
    baseline3["method"] = "Baseline 3"
    baseline4["method"] = "Baseline 4"
    ahead = pd.concat([baseline3, baseline4], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    sns.lineplot(data=ahead, x="ahead_days", y="mean_abs_rel_bias_pct", hue="method", marker="o", linewidth=2.0, ax=axes[0])
    sns.lineplot(data=ahead, x="ahead_days", y="pearson_r", hue="method", marker="o", linewidth=2.0, ax=axes[1])
    axes[0].set_title("Lead-day absolute relative error")
    axes[0].set_xlabel("Lead day")
    axes[0].set_ylabel("Mean absolute relative error (%)")
    axes[1].set_title("Lead-day correlation skill")
    axes[1].set_xlabel("Lead day")
    axes[1].set_ylabel("Pearson R")
    return save_figure(fig, output_root / "point_forecast" / "actual_forecast_lead_day_error.png")


def render_public_figures(source_root: Path, project_root: Path) -> list[Path]:
    configure_plot_style()
    output_root = project_root / "figures"
    metrics = read_csv_checked(project_root / "results_public" / "public_metrics_summary.csv", ["method", "mape_percent", "bias_percent"], "公开指标汇总")
    monthly = read_csv_checked(project_root / "results_public" / "public_monthly_predictions_index.csv", ["method", "month", "energy_true_index", "energy_pred_index", "error_percent"], "公开逐月归一化预测")
    curves = read_power_curve_candidates(source_root)

    return [
        render_wind_distribution(source_root, output_root),
        render_daily_wind_energy(source_root, output_root),
        render_power_curve_all_turbines(curves, output_root),
        render_power_curve_seasonal(curves, output_root),
        render_metrics(metrics, output_root),
        render_monthly_index(monthly, output_root),
        render_error_heatmap(monthly, output_root),
        render_lead_day_error(source_root, output_root),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render public-safe figures from local private baseline outputs.")
    parser.add_argument("--source-root", required=True, help="Local private baseline workspace root. The value is not written into public files.")
    parser.add_argument("--project-root", required=True, help="Public repository root where figures are written.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = require_directory(Path(args.source_root), "本地私有基线工作区")
    project_root = require_directory(Path(args.project_root), "公开仓库根目录")
    output_paths = render_public_figures(source_root, project_root)
    for output_path in output_paths:
        print(f"{output_path.relative_to(project_root)} {output_path.stat().st_size}")


if __name__ == "__main__":
    main()
