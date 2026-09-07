from pathlib import Path
import json
import sys
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wind_power_baselines.baselines import build_all_baselines
from wind_power_baselines.evaluation import evaluate_predictions, normalize_public_predictions
from wind_power_baselines.power_curve import build_power_curve
from wind_power_baselines.preprocessing import clean_scada, prepare_forecast


class PublicWorkflowTest(unittest.TestCase):
    def test_scada_cleaning_removes_invalid_rows_without_mutating_input(self) -> None:
        raw = pd.DataFrame(
            {
                "timestamp": ["2025-01-01 00:00", "bad-time", "2025-01-01 00:15"],
                "turbine_id": ["T01", "T01", "T01"],
                "wind_speed": [5.0, 6.0, -1.0],
                "power_kw": [120.0, 130.0, 20.0],
                "status": ["normal", "normal", "normal"],
            }
        )

        cleaned = clean_scada(raw)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(raw), 3)
        self.assertIn("season", cleaned.columns)

    def test_power_curve_uses_bin_medians_and_sample_counts(self) -> None:
        scada = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=5, freq="15min"),
                "turbine_id": ["T01", "T01", "T01", "T01", "T01"],
                "wind_speed": [4.1, 4.2, 4.3, 8.0, 8.2],
                "power_kw": [80.0, 90.0, 1000.0, 430.0, 450.0],
                "status": ["normal", "normal", "normal", "normal", "normal"],
                "season": ["winter", "winter", "winter", "winter", "winter"],
            }
        )

        curve = build_power_curve(scada, bin_width=1.0, min_samples=2)

        self.assertEqual(set(curve.columns), {"turbine_id", "season", "wind_speed_bin", "power_kw", "sample_count"})
        self.assertGreaterEqual(curve["sample_count"].min(), 2)

    def test_baseline_outputs_and_metrics_have_public_shape(self) -> None:
        monthly_truth = pd.DataFrame(
            {
                "month": ["2024-01", "2024-02", "2025-01", "2025-02"],
                "energy_true": [100.0, 120.0, 110.0, 150.0],
            }
        )
        forecast = prepare_forecast(
            pd.DataFrame(
                {
                    "issue_time": ["2024-12-16", "2024-12-16", "2025-01-16", "2025-01-16"],
                    "target_time": ["2025-01-01", "2025-01-02", "2025-02-01", "2025-02-02"],
                    "lead_day": [16, 17, 16, 17],
                    "wind_speed": [5.0, 6.0, 7.0, 8.0],
                }
            )
        )
        power_curve = pd.DataFrame(
            {
                "turbine_id": ["T01", "T01", "T01"],
                "season": ["winter", "winter", "winter"],
                "wind_speed_bin": [5.0, 6.0, 7.0],
                "power_kw": [100.0, 180.0, 280.0],
                "sample_count": [10, 10, 10],
            }
        )
        correction = pd.DataFrame(
            {
                "month": ["2025-01", "2025-02"],
                "correction_factor": [0.95, 1.05],
            }
        )

        predictions = build_all_baselines(monthly_truth, forecast, power_curve, correction)
        metrics = evaluate_predictions(predictions)
        public_rows = normalize_public_predictions(predictions)

        self.assertEqual(set(predictions["method"]), {"Baseline 1", "Baseline 2", "Baseline 3", "Baseline 4", "AK-D"})
        self.assertTrue({"mape_percent", "bias_percent", "pearson_r"}.issubset(metrics.columns))
        self.assertTrue({"energy_true_index", "energy_pred_index"}.issubset(public_rows.columns))
        self.assertFalse(public_rows.to_csv(index=False).find(":\\") >= 0)

    def test_public_figure_set_contains_real_result_story_without_raw_data(self) -> None:
        expected_figure_paths = [
            PROJECT_ROOT / "figures" / "power_curve" / "actual_wind_distribution_by_turbine.png",
            PROJECT_ROOT / "figures" / "power_curve" / "actual_daily_wind_energy_scatter.png",
            PROJECT_ROOT / "figures" / "power_curve" / "actual_power_curve_all_turbines.png",
            PROJECT_ROOT / "figures" / "power_curve" / "actual_power_curve_seasonal.png",
            PROJECT_ROOT / "figures" / "point_forecast" / "actual_baseline_mape_bias.png",
            PROJECT_ROOT / "figures" / "point_forecast" / "actual_monthly_prediction_index.png",
            PROJECT_ROOT / "figures" / "point_forecast" / "actual_monthly_error_heatmap.png",
            PROJECT_ROOT / "figures" / "point_forecast" / "actual_forecast_lead_day_error.png",
        ]

        for figure_path in expected_figure_paths:
            with self.subTest(figure_path=figure_path):
                self.assertTrue(figure_path.exists(), f"缺少公开高清图: path={figure_path}")
                self.assertGreater(figure_path.stat().st_size, 20_000, f"公开图可能未正常生成: path={figure_path}")

        forbidden_suffixes = {".xlsx", ".xls", ".parquet", ".pkl"}
        committed_files = [path for path in PROJECT_ROOT.rglob("*") if ".git" not in path.parts and path.is_file()]
        leaked_raw_files = [path for path in committed_files if path.suffix.lower() in forbidden_suffixes]

        self.assertEqual(leaked_raw_files, [])

    def test_notebooks_reference_public_high_resolution_figures(self) -> None:
        notebook_text = "\n".join(
            [
                (PROJECT_ROOT / "notebooks" / "01_power_curve_fitting.ipynb").read_text(encoding="utf-8"),
                (PROJECT_ROOT / "notebooks" / "02_point_forecast_baselines.ipynb").read_text(encoding="utf-8"),
            ]
        )
        expected_references = [
            "../figures/power_curve/actual_wind_distribution_by_turbine.png",
            "../figures/power_curve/actual_daily_wind_energy_scatter.png",
            "../figures/power_curve/actual_power_curve_all_turbines.png",
            "../figures/power_curve/actual_power_curve_seasonal.png",
            "../figures/point_forecast/actual_baseline_mape_bias.png",
            "../figures/point_forecast/actual_monthly_prediction_index.png",
            "../figures/point_forecast/actual_monthly_error_heatmap.png",
            "../figures/point_forecast/actual_forecast_lead_day_error.png",
        ]

        for reference in expected_references:
            with self.subTest(reference=reference):
                self.assertIn(reference, notebook_text)

        for notebook_path in [
            PROJECT_ROOT / "notebooks" / "01_power_curve_fitting.ipynb",
            PROJECT_ROOT / "notebooks" / "02_point_forecast_baselines.ipynb",
        ]:
            with self.subTest(notebook_path=notebook_path):
                notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)


if __name__ == "__main__":
    unittest.main()
