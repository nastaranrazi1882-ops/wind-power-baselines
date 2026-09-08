from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from wind_power_baselines.evaluation import evaluate_predictions
from wind_power_baselines.plotting import plot_metric_bars, plot_power_curve
from wind_power_baselines.power_curve import build_power_curve
from wind_power_baselines.preprocessing import clean_scada, prepare_forecast


def ensure_directories(project_root: Path) -> None:
    for path in [
        project_root / "data_sample",
        project_root / "results_public",
        project_root / "figures",
        project_root / "notebooks",
        project_root / "scripts",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def synthetic_power(wind_speed: np.ndarray, rated_power: float) -> np.ndarray:
    cubic = rated_power * np.clip((wind_speed - 2.5) / 9.5, 0.0, 1.0) ** 3
    plateau = np.where(wind_speed >= 12.0, rated_power, cubic)
    return np.where(wind_speed >= 25.0, 0.0, plateau)


def create_sample_scada(project_root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(20260907)
    timestamps = pd.date_range("2025-01-01", periods=2 * 96, freq="15min")
    frames: list[pd.DataFrame] = []
    for turbine_index, turbine_id in enumerate(["T01", "T02", "T03"]):
        hour_angle = 2.0 * math.pi * timestamps.hour.to_numpy() / 24.0
        day_angle = 2.0 * math.pi * timestamps.dayofyear.to_numpy() / 365.0
        base_wind = 6.0 + 1.0 * np.sin(day_angle) + 0.35 * np.cos(hour_angle) + turbine_index * 0.15
        wind_speed = np.clip(base_wind + rng.normal(0.0, 1.1, len(timestamps)), 0.0, 24.0)
        power_kw = synthetic_power(wind_speed, 2200.0) + rng.normal(0.0, 55.0, len(timestamps))
        status = np.where(rng.random(len(timestamps)) < 0.97, "normal", "fault")
        frame = pd.DataFrame(
            {
                "timestamp": timestamps.astype(str),
                "turbine_id": turbine_id,
                "wind_speed": np.round(wind_speed, 3),
                "power_kw": np.round(np.maximum(power_kw, 0.0), 3),
                "status": status,
            }
        )
        frames.append(frame)
    sample = pd.concat(frames, ignore_index=True)
    injected = pd.DataFrame(
        {
            "timestamp": ["not-a-time", "2025-01-02 00:00:00", "2025-01-03 00:00:00"],
            "turbine_id": ["T01", "T01", "T02"],
            "wind_speed": [5.0, -2.0, 45.0],
            "power_kw": [100.0, 50.0, -10.0],
            "status": ["normal", "normal", "normal"],
        }
    )
    output = pd.concat([sample, injected], ignore_index=True)
    output.to_csv(project_root / "data_sample" / "sample_turbine_scada.csv", index=False, encoding="utf-8")
    return output


def create_sample_monthly_truth(project_root: Path) -> pd.DataFrame:
    months = pd.period_range("2024-01", "2025-06", freq="M").astype(str)
    month_numbers = pd.to_datetime(pd.Series(months) + "-01").dt.month.to_numpy()
    seasonal = 1120.0 + 260.0 * np.cos((month_numbers - 1.0) / 12.0 * 2.0 * math.pi)
    trend = np.linspace(-40.0, 70.0, len(months))
    truth = pd.DataFrame({"month": months, "energy_true": np.round(seasonal + trend, 3)})
    truth.to_csv(project_root / "data_sample" / "sample_monthly_truth.csv", index=False, encoding="utf-8")
    return truth


def create_sample_forecast(project_root: Path) -> pd.DataFrame:
    rng = np.random.default_rng(20260908)
    rows: list[dict[str, object]] = []
    for target_month in pd.period_range("2025-01", "2025-06", freq="M"):
        issue_time = (target_month.to_timestamp() - pd.DateOffset(days=16)).normalize()
        for target_time in pd.date_range(target_month.to_timestamp(), target_month.to_timestamp() + pd.offsets.MonthEnd(0), freq="D"):
            lead_day = int((target_time.normalize() - issue_time).days)
            month_factor = 5.8 + 1.1 * math.cos((target_month.month - 1.0) / 12.0 * 2.0 * math.pi)
            wind_speed = max(0.0, month_factor + rng.normal(0.0, 1.0))
            rows.append(
                {
                    "issue_time": issue_time.date().isoformat(),
                    "target_time": target_time.date().isoformat(),
                    "lead_day": lead_day,
                    "wind_speed": round(wind_speed, 3),
                }
            )
    forecast = pd.DataFrame(rows)
    forecast.to_csv(project_root / "data_sample" / "sample_weather_forecast.csv", index=False, encoding="utf-8")
    return forecast


def create_sample_correction(project_root: Path) -> pd.DataFrame:
    months = pd.period_range("2025-01", "2025-06", freq="M").astype(str)
    factors = [0.92, 0.98, 1.04, 1.06, 1.02, 0.96]
    correction = pd.DataFrame({"month": months, "correction_factor": factors})
    correction.to_csv(project_root / "data_sample" / "sample_ak_monthly_correction.csv", index=False, encoding="utf-8")
    return correction


def create_public_power_curve(project_root: Path, raw_scada: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_scada(raw_scada)
    curve = build_power_curve(cleaned, 0.5, 12)
    curve.to_csv(project_root / "data_sample" / "sample_power_curve.csv", index=False, encoding="utf-8")
    plot_power_curve(curve, project_root / "figures" / "sample_power_curve.png")
    return curve


def create_sample_predictions(project_root: Path, truth: pd.DataFrame, forecast: pd.DataFrame, curve: pd.DataFrame, correction: pd.DataFrame) -> pd.DataFrame:
    from wind_power_baselines.baselines import build_all_baselines

    prepared = prepare_forecast(forecast)
    predictions = build_all_baselines(truth, prepared, curve, correction)
    predictions.to_csv(project_root / "data_sample" / "sample_monthly_predictions.csv", index=False, encoding="utf-8")
    return predictions


def read_public_metrics(project_root: Path) -> pd.DataFrame:
    path = project_root / "results_public" / "public_metrics_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"找不到脱敏公开指标文件: path={path}")
    return pd.read_csv(path)


def markdown_cell(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(code: str) -> dict[str, object]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    normalized_cells: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        copied = dict(cell)
        copied["id"] = f"cell-{index:03d}"
        normalized_cells.append(copied)
    return {
        "cells": normalized_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=2), encoding="utf-8")


def create_power_curve_notebook(project_root: Path) -> None:
    cells = [
        markdown_cell("# 01 Power Curve Fitting\n\n这个 notebook 展示风机级 SCADA 样例数据如何清洗、预处理并构建经验功率曲线。公开版只使用合成样例数据，字段和处理流程对应真实工程，但不包含真实风机原始数据。"),
        markdown_cell("## 1. Data Description\n\n`sample_turbine_scada.csv` 包含 `timestamp`、`turbine_id`、`wind_speed`、`power_kw`、`status`。真实项目中这些字段通常来自 15 分钟 SCADA 报表；公开样例保留字段结构，用于说明功率曲线拟合过程。"),
        code_cell("from pathlib import Path\nimport sys\n\nPROJECT_ROOT = Path.cwd()\nif not (PROJECT_ROOT / 'src').exists():\n    PROJECT_ROOT = PROJECT_ROOT.parent\nsys.path.insert(0, str(PROJECT_ROOT / 'src'))\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\n\nfrom wind_power_baselines.preprocessing import clean_scada\nfrom wind_power_baselines.power_curve import build_power_curve\n\nsns.set_theme(style='whitegrid')\nscada = pd.read_csv(PROJECT_ROOT / 'data_sample' / 'sample_turbine_scada.csv')\nscada.head()"),
        markdown_cell("## 2. Data Cleaning\n\n清洗步骤包括：解析时间、删除重复时间戳、保留 `normal` 状态、剔除负风速/超物理风速、剔除负功率。这样做的原因是停机、限功率和采集异常会把经验功率曲线拉偏。"),
        code_cell("cleaned = clean_scada(scada)\nprint({'raw_rows': len(scada), 'cleaned_rows': len(cleaned), 'removed_rows': len(scada) - len(cleaned)})\ncleaned.head()"),
        markdown_cell("## 3. EDA Before Fitting\n\n先看风速和功率的散点分布，再看不同风机的样本覆盖。真实项目中如果某台风机某个风速段样本很少，曲线会不稳定，需要在评估里标注。"),
        code_cell("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\nsns.scatterplot(data=cleaned.sample(min(1000, len(cleaned)), random_state=7), x='wind_speed', y='power_kw', hue='turbine_id', s=12, ax=axes[0])\nsns.countplot(data=cleaned, x='turbine_id', hue='season', ax=axes[1])\naxes[0].set_title('Cleaned wind speed vs power')\naxes[1].set_title('Sample coverage by turbine and season')\nfig.tight_layout()"),
        markdown_cell("## 4. Preprocessing And Curve Construction\n\n公开版采用风速分箱后的中位数功率作为经验功率曲线。中位数比均值更稳健，因为散点里常有停机、限功率、切出和传感器异常。"),
        code_cell("curve = build_power_curve(cleaned, 0.5, 12)\ncurve.to_csv(PROJECT_ROOT / 'data_sample' / 'sample_power_curve.csv', index=False)\ncurve.head()"),
        markdown_cell("## 5. Fitting Result\n\n下图展示每台风机、每个季节的经验功率曲线。第二个点预测 notebook 会直接读取这个曲线表，把预报风速映射为电量。"),
        code_cell("fig, ax = plt.subplots(figsize=(9, 5))\nfor (turbine_id, season), group in curve.groupby(['turbine_id', 'season']):\n    ax.plot(group['wind_speed_bin'], group['power_kw'], marker='o', label=f'{turbine_id}-{season}')\nax.set_xlabel('Wind speed bin (m/s)')\nax.set_ylabel('Median power (kW)')\nax.set_title('Empirical power curves from cleaned SCADA sample')\nax.legend(fontsize=8)\nfig.tight_layout()"),
        markdown_cell("## 6. Actual Public Figures And Conclusions\n\n下面几张图由本地真实中间结果渲染后以 PNG 形式放入仓库。原始 SCADA 和原始风塔/测风文件不上传，但图可以正常展示真实分布和拟合形态。\n\n![Observed wind speed distribution](../figures/power_curve/actual_wind_distribution_by_turbine.png)\n\n**结论。** 6 台风机的观测风速分布整体接近，均值大约落在 5.76-5.91 m/s，说明场内各机组处在相近风况下，适合做统一口径的场站级基线评估。T06 的样本数和均值略低，提示它可能存在更多缺测或局部风况差异；因此功率曲线既要看场站平均，也要保留逐机检查。\n\n![Daily observed wind speed vs normalized energy](../figures/power_curve/actual_daily_wind_energy_scatter.png)\n\n**结论。** 日均观测风速和日电量指数呈明显正相关，相关系数约 0.894。低风速段电量变化较平缓，中高风速段电量上升更快，符合风机功率曲线的非线性特征。这说明 Baseline 2/3/4 和 AK-D 不能只做电量均值外推，必须把风速到功率的映射建好。\n\n![Actual fitted power curves by turbine](../figures/power_curve/actual_power_curve_all_turbines.png)\n\n**结论。** 左图中曲线大量重叠，不代表“没有信息”，而是说明按各风机自身最大功率归一化后，6 台风机的主工作区间形态基本一致：约 3-10 m/s 为爬坡段，10 m/s 以上进入平台段。右图把每条曲线减去同季节、同风速 bin 的场站中位曲线后，可以看到差异主要集中在爬坡段和高风速尾部；这些区域正是月度预测误差最容易被放大的地方。\n\n![Seasonal fitted power curves](../figures/power_curve/actual_power_curve_seasonal.png)\n\n**结论。** 左图显示冷暖季曲线在爬坡段有可见差异，因此分季节建曲线是合理的。右图已经改为真实观测样本数，而不是“有曲线点的风机数量”：样本主要集中在常见风速区间，高风速 bin 样本显著变少，所以高风速尾部即使画出了曲线，也应当在解释中降低置信度。"),
        markdown_cell("## 7. Output\n\n输出文件是 `data_sample/sample_power_curve.csv`，字段为 `turbine_id`、`season`、`wind_speed_bin`、`power_kw`、`sample_count`。它是后续 Baseline 2/3/4 和 AK-D 的风速到功率映射底座。"),
    ]
    write_notebook(project_root / "notebooks" / "01_power_curve_fitting.ipynb", cells)


def create_baseline_notebook(project_root: Path) -> None:
    cells = [
        markdown_cell("# 02 Point Forecast Baselines\n\n这个 notebook 整理最终 5 条月度点预测基线：Baseline 1、Baseline 2、Baseline 3、Baseline 4 和 AK-D。公开版使用合成样例数据跑通流程，并用脱敏真实指标做结果分析。"),
        markdown_cell("## 1. Data Description\n\n输入包括月度真值、气象预报、功率曲线和 AK-D 分月订正系数。真实项目中的 EC45、ERA5、SCADA 和月度理论电量不会上传；公开样例只保留字段结构。"),
        code_cell("from pathlib import Path\nimport sys\n\nPROJECT_ROOT = Path.cwd()\nif not (PROJECT_ROOT / 'src').exists():\n    PROJECT_ROOT = PROJECT_ROOT.parent\nsys.path.insert(0, str(PROJECT_ROOT / 'src'))\n\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\n\nfrom wind_power_baselines.baselines import build_all_baselines\nfrom wind_power_baselines.evaluation import evaluate_predictions, normalize_public_predictions\nfrom wind_power_baselines.preprocessing import clean_monthly_truth, prepare_forecast\n\nsns.set_theme(style='whitegrid')\ntruth = pd.read_csv(PROJECT_ROOT / 'data_sample' / 'sample_monthly_truth.csv')\nforecast_raw = pd.read_csv(PROJECT_ROOT / 'data_sample' / 'sample_weather_forecast.csv')\npower_curve = pd.read_csv(PROJECT_ROOT / 'data_sample' / 'sample_power_curve.csv')\ncorrection = pd.read_csv(PROJECT_ROOT / 'data_sample' / 'sample_ak_monthly_correction.csv')\ntruth.head(), forecast_raw.head(), power_curve.head(), correction.head()"),
        markdown_cell("## 2. Cleaning And Preprocessing\n\n这里完成月份标准化、时间解析、风速合法性检查、预测提前期检查和预测月份对齐。公开样例故意保持简单，重点是展示真实工程里需要固定的数据契约。"),
        code_cell("truth_clean = clean_monthly_truth(truth)\nforecast = prepare_forecast(forecast_raw)\nprint({'truth_months': len(truth_clean), 'forecast_rows': len(forecast), 'forecast_months': forecast['month'].nunique()})\nforecast.head()"),
        markdown_cell("## 3. EDA\n\nEDA 先回答三个问题：真值有没有季节性，预报风速覆盖哪些月份，功率曲线是否能覆盖样例风速范围。"),
        code_cell("fig, axes = plt.subplots(1, 3, figsize=(15, 4))\nsns.lineplot(data=truth_clean, x='month', y='energy_true', marker='o', ax=axes[0])\nsns.boxplot(data=forecast, x='month', y='wind_speed', ax=axes[1])\nsns.lineplot(data=power_curve, x='wind_speed_bin', y='power_kw', hue='season', estimator='mean', errorbar=None, ax=axes[2])\naxes[0].tick_params(axis='x', rotation=45)\naxes[1].tick_params(axis='x', rotation=45)\naxes[0].set_title('Monthly truth sample')\naxes[1].set_title('Forecast wind speed by month')\naxes[2].set_title('Power curve coverage')\nfig.tight_layout()"),
        markdown_cell("## 4. Baseline Methods\n\n- Baseline 1：历史同期电量均值，作为无气象信息的朴素基线。\n- Baseline 2：ERA5/气候库思想，先用功率曲线得到历史电量库，再取 P50。\n- Baseline 3：EC45 + LLS 订正 + 功率曲线，强调中长期预报风速到电量的映射。\n- Baseline 4：EC45 + 相似日/覆盖优选 + 功率曲线，强调从历史相似状态借用电量水平。\n- AK-D：分月订正 AK 方法，把订正系数按月份拆开，用来缓解全年同一订正系数导致的小风月高估和大风月低估。"),
        code_cell("predictions = build_all_baselines(truth_clean, forecast, power_curve, correction)\npredictions"),
        markdown_cell("## 5. Evaluation\n\n统一计算 MAPE、汇总偏差和 PearsonR。真实评估中还会按 W1/W2/W3 等窗口拆分；公开样例只演示接口，真实结果在 `results_public` 中以脱敏形式给出。"),
        code_cell("metrics = evaluate_predictions(predictions)\npublic_predictions = normalize_public_predictions(predictions)\nmetrics, public_predictions.head()"),
        markdown_cell("## 6. Public Result Analysis\n\n下面读取真实工程脱敏后的结果。这里不展示真实 kWh/GWh，只展示每个方法在自身可用月份上的 MAPE、Bias、PearsonR，以及逐月归一化指数。"),
        code_cell("public_metrics = pd.read_csv(PROJECT_ROOT / 'results_public' / 'public_metrics_summary.csv')\npublic_monthly = pd.read_csv(PROJECT_ROOT / 'results_public' / 'public_monthly_predictions_index.csv')\npublic_metrics"),
        code_cell("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\nsns.barplot(data=public_metrics, x='method', y='mape_percent', ax=axes[0])\nsns.barplot(data=public_metrics, x='method', y='bias_percent', ax=axes[1])\nfor axis in axes:\n    axis.tick_params(axis='x', rotation=30)\naxes[0].set_title('Public MAPE by method')\naxes[1].set_title('Public bias by method')\nfig.tight_layout()"),
        code_cell("plot_monthly = public_monthly.copy()\nplot_monthly['month_date'] = pd.PeriodIndex(plot_monthly['month'].astype(str), freq='M').to_timestamp()\nplot_monthly = plot_monthly.sort_values(['month_date', 'method'])\nfig, ax = plt.subplots(figsize=(11, 5))\nsns.lineplot(data=plot_monthly, x='month_date', y='energy_pred_index', hue='method', marker='o', ax=ax)\nsns.lineplot(data=plot_monthly.drop_duplicates('month').sort_values('month_date'), x='month_date', y='energy_true_index', color='black', marker='o', label='Truth index', ax=ax)\nax.tick_params(axis='x', rotation=45)\nax.set_title('Normalized monthly predictions and truth')\nax.set_xlabel('month')\nax.set_ylabel('Index, truth-window mean = 1')\nfig.tight_layout()"),
        markdown_cell("## 7. Actual Public Figures And Conclusions\n\n下面几张图由真实评估结果和真实中间指标渲染，公开仓库只保留 PNG 与脱敏指标，不保留原始电量表、风塔表或本地路径。\n\n![Baseline MAPE and bias](../figures/point_forecast/actual_baseline_mape_bias.png)\n\n**结论。** Baseline 1 的 MAPE 约 13.73%，在当前同窗口样本上是最稳的历史参照；Baseline 2 的相关性接近 Baseline 1，但 MAPE 更高，说明 P50/气候库路线在部分月份会偏离真实月度状态。Baseline 3 的整体 Bias 较小，但相关性不高，表示它的总量校正有帮助，逐月峰谷仍不够稳。Baseline 4 在当前口径下 MAPE 和相关性都较弱，说明相似日筛选没有稳定转化成月度优势。AK-D 的 Bias 接近 0，但 MAPE 不最低，说明分月订正改善了总偏差，却仍会在个别月份过修正。\n\n![Monthly prediction index](../figures/point_forecast/actual_monthly_prediction_index.png)\n\n**结论。** 这张图已经改成真实日期轴，按 2023-09 到 2026-03 顺序排列。之前 AK-D 看起来在 2025-12 到 2024-07 之间像一条平线，是因为月份被当成字符串分类轴，显示顺序跟 CSV 行顺序走，视觉上把不同年份错连了；不是 AK-D 真实预测为平。修正后可以看到 AK-D 在 2024-07 到 2026-01 连续变化，2024-10 附近明显高估，2025-02/03 明显低估，2025-11/12 又低于真值高峰。\n\n![Monthly relative error heatmap](../figures/point_forecast/actual_monthly_error_heatmap.png)\n\n**结论。** 热力图显示 Baseline 4 在 2024-07、2024-08、2025-08 等小风或异常月份高估最明显，说明相似日方法对极端低发月份不够鲁棒。Baseline 3 也在 2024-07、2025-06 等月份高估，说明 EC45 + LLS + 功率曲线仍会受风速订正和月内覆盖影响。AK-D 的误差颜色更分散，总偏差较小，但 2024-08 高估和 2025-02/03 低估说明分月订正不能单独解决所有异常月。\n\n![Lead-day error and correlation](../figures/point_forecast/actual_forecast_lead_day_error.png)\n\n**结论。** 提前期图只对 Baseline 3/4 展开，因为它们依赖 EC45 起报和提前期。Baseline 3 的误差随提前期增加整体抬升，相关性也下降，说明 LLS 订正对较近预报更有效；Baseline 4 的短提前期相关性较好，但误差水平不低，说明相似日筛选能抓到部分趋势，却容易在电量尺度上偏离。最终月度点预测要同时看提前期、月内覆盖和功率曲线映射，不能只看单日预报技巧。"),
        markdown_cell("## 8. Conclusion\n\n从真实脱敏结果看，5 条最终点预测基线覆盖了从朴素历史同期、气候库功率曲线、EC45 订正、相似日，到 AK-D 分月订正的完整路线。AK-D 的价值是说明订正系数按月份拆分后，部分极端月份误差会回落；它仍需要关注分月样本少导致的过修正。"),
    ]
    write_notebook(project_root / "notebooks" / "02_point_forecast_baselines.ipynb", cells)


def create_readme(project_root: Path, metrics: pd.DataFrame) -> None:
    metrics_markdown = metrics[["method", "month_count", "mape_percent", "bias_percent", "pearson_r"]].to_markdown(index=False)
    readme = f"""# Wind Power Baselines

This repository is a public, data-safe walkthrough of wind power monthly point forecasting baselines.

## What Is Included

- `notebooks/01_power_curve_fitting.ipynb`: SCADA-style sample data cleaning, preprocessing, empirical power curve fitting, and fitting diagnostics.
- `notebooks/02_point_forecast_baselines.ipynb`: final point forecasting baselines: Baseline 1, Baseline 2, Baseline 3, Baseline 4, and AK-D.
- `src/wind_power_baselines/`: small functional utilities for cleaning, preprocessing, power curve construction, baseline prediction, metrics, and plots.
- `data_sample/`: synthetic sample data only.
- `results_public/`: anonymized real evaluation results. Absolute energy values are not published.
- `figures/`: public-safe PNG figures rendered from sample data, anonymized real metrics, and local real intermediate outputs.

## Data Safety

The real project used private wind farm data, raw SCADA files, forecast files, and internal paths. Those files are intentionally excluded.

Public results use normalized indices and percentages:

- `energy_true_index`
- `energy_pred_index`
- `mape_percent`
- `bias_percent`
- `pearson_r`

No original kWh/GWh series, local drive paths, credentials, or raw station files are included.

Raw SCADA and raw mast/met-tower tables are not committed. Some figures are rendered from local real intermediate outputs and then published as PNG files so the notebooks can show realistic distributions, fitted curves, monthly errors, and lead-day behavior without exposing source tables.

## Figure Gallery

### Power Curve And Observed Wind

![Observed wind speed distribution](figures/power_curve/actual_wind_distribution_by_turbine.png)

![Daily observed wind speed vs normalized energy](figures/power_curve/actual_daily_wind_energy_scatter.png)

![Actual fitted power curves by turbine](figures/power_curve/actual_power_curve_all_turbines.png)

![Seasonal fitted power curves](figures/power_curve/actual_power_curve_seasonal.png)

### Point Forecast Evaluation

![Baseline MAPE and bias](figures/point_forecast/actual_baseline_mape_bias.png)

![Monthly prediction index](figures/point_forecast/actual_monthly_prediction_index.png)

![Monthly relative error heatmap](figures/point_forecast/actual_monthly_error_heatmap.png)

![Lead-day error and correlation](figures/point_forecast/actual_forecast_lead_day_error.png)

## Notebook Reading Order

1. Open `notebooks/01_power_curve_fitting.ipynb` to understand how turbine-level data becomes empirical power curves.
2. Open `notebooks/02_point_forecast_baselines.ipynb` to see how those curves support monthly point forecasting baselines.

## Quick Start

```powershell
python -m pip install -e .
python -m unittest discover -s tests
jupyter notebook notebooks
```

## Final Baselines

| Method | Description |
| --- | --- |
| Baseline 1 | Historical same-month energy mean |
| Baseline 2 | ERA5/climatology idea + seasonal power curve + P50 |
| Baseline 3 | EC45 + LLS correction + power curve |
| Baseline 4 | EC45 + similar-day / coverage selection + power curve |
| AK-D | Monthly AK correction, keeping the AK route but fitting correction by month |

## Public Metrics Summary

{metrics_markdown}

## Repository Boundary

This repository documents the baseline construction and evaluation process. It does not publish private source data, internal file paths, or the full private reproduction archive.
"""
    (project_root / "README.md").write_text(readme, encoding="utf-8")


def create_gitignore(project_root: Path) -> None:
    lines = [
        "# Python",
        "__pycache__/",
        "*.py[cod]",
        ".pytest_cache/",
        ".mypy_cache/",
        ".ruff_cache/",
        ".ipython/",
        ".jupyter_config/",
        ".jupyter_data/",
        ".jupyter_runtime/",
        "",
        "# Local environments",
        "." + "env",
        ".venv/",
        "venv/",
        "*.key",
        "*.pem",
        "",
        "# Raw data",
        "data/",
        "raw/",
        "private/",
        "*.xlsx",
        "*.xls",
        "*.parquet",
        "*.pkl",
        "",
        "# Large/local outputs",
        "outputs/",
        "cache/",
        "*.log",
    ]
    text = "\n".join(lines) + "\n"
    (project_root / ".gitignore").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_directories(PROJECT_ROOT)
    raw_scada = create_sample_scada(PROJECT_ROOT)
    truth = create_sample_monthly_truth(PROJECT_ROOT)
    forecast = create_sample_forecast(PROJECT_ROOT)
    correction = create_sample_correction(PROJECT_ROOT)
    curve = create_public_power_curve(PROJECT_ROOT, raw_scada)
    create_sample_predictions(PROJECT_ROOT, truth, forecast, curve, correction)
    metrics = read_public_metrics(PROJECT_ROOT)
    plot_metric_bars(metrics, PROJECT_ROOT / "figures" / "public_metrics_summary.png")
    create_power_curve_notebook(PROJECT_ROOT)
    create_baseline_notebook(PROJECT_ROOT)
    create_readme(PROJECT_ROOT, metrics)
    create_gitignore(PROJECT_ROOT)


if __name__ == "__main__":
    main()
