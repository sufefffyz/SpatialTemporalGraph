import importlib.util
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
UTILS_PATH = (
    REPO_ROOT
    / "mvp_experiments"
    / "active"
    / "adaptive_graph_benchmark"
    / "scripts"
    / "largest_style_metrics_utils.py"
)


def load_utils():
    spec = importlib.util.spec_from_file_location("largest_style_metrics_utils", UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LargestStyleMetricsTest(unittest.TestCase):
    def test_largest_average_uses_horizon_simple_mean(self):
        utils = load_utils()
        metrics = {
            "overall": {"MAE": 999.0, "RMSE": 999.0, "MAPE": 999.0},
            "horizon_1": {"MAE": 1.0, "RMSE": 10.0, "MAPE": 0.1},
            "horizon_2": {"MAE": 3.0, "RMSE": 20.0, "MAPE": 0.2},
            "horizon_3": {"MAE": 5.0, "RMSE": 30.0, "MAPE": 0.3},
        }

        average = utils.compute_largest_average(metrics, horizons=[1, 2, 3])

        self.assertEqual(average["MAE"], 3.0)
        self.assertEqual(average["RMSE"], 20.0)
        self.assertAlmostEqual(average["MAPE"], 0.2)

    def test_summary_cell_formats_three_metrics(self):
        utils = load_utils()

        cell = utils.format_summary_cell({"MAE": 1.23456, "RMSE": 7.89012, "MAPE": 0.34567})

        self.assertEqual(cell, "1.2346 / 7.8901 / 0.3457")


if __name__ == "__main__":
    unittest.main()
