import json
import tempfile
import unittest
from pathlib import Path

from tools.private_proxy_score import (
    CalibrationPoint,
    estimate_private,
    fit_rus_gap_calibration,
    grouped_cv_stats,
    load_validation_metrics,
)


POINTS = [
    CalibrationPoint("soup4_relf", "soup", 0.9939318717794705, 0.99034, 0.9898469170474977),
    CalibrationPoint("uniform16", "uniform", 0.9913425122836846, 0.99085, 0.9862350484118102),
    CalibrationPoint("uniform16_hf", "uniform", 0.9930211244131191, 0.99282, 0.9862350484118102),
    CalibrationPoint("gate987", "gate", 0.9939838896049196, 0.98986, 0.9908948636778465),
    CalibrationPoint("gate987_hf", "gate", 0.9951574114150464, 0.99182, 0.9908948636778465),
]


class PrivateProxyScoreTests(unittest.TestCase):
    def test_fits_documented_rus_gap_relationship(self):
        model = fit_rus_gap_calibration(POINTS)
        self.assertAlmostEqual(model.center, 0.98882135, places=7)
        self.assertAlmostEqual(model.intercept, -0.00234936, places=7)
        self.assertAlmostEqual(model.slope, -0.753883, places=5)

    def test_estimate_reproduces_uniform_headfit_within_empirical_error(self):
        model = fit_rus_gap_calibration(POINTS)
        result = estimate_private(
            local_final=0.9930211244131191,
            rus_o=0.9862350484118102,
            cka_f_o=0.013237623311194498,
            model=model,
        )
        self.assertAlmostEqual(result["private_proxy_v1"], 0.99262, places=5)
        self.assertTrue(result["guardrail_pass"])
        self.assertFalse(result["rus_extrapolated"])

    def test_clamps_rus_outside_calibration_range_and_marks_extrapolation(self):
        model = fit_rus_gap_calibration(POINTS)
        result = estimate_private(0.994, 0.999, 0.01, model)
        edge = estimate_private(0.994, model.rus_max, 0.01, model)
        self.assertTrue(result["rus_extrapolated"])
        self.assertAlmostEqual(result["private_proxy_v1"], edge["private_proxy_v1"])

    def test_hard_guardrail_blocks_ranking_score(self):
        model = fit_rus_gap_calibration(POINTS)
        result = estimate_private(0.994, 0.989, 0.031, model)
        self.assertFalse(result["guardrail_pass"])
        self.assertIsNone(result["selection_score"])

    def test_grouped_cross_validation_matches_recorded_audit(self):
        stats = grouped_cv_stats(POINTS)
        self.assertEqual(stats["n_points"], 5)
        self.assertEqual(stats["n_families"], 3)
        self.assertAlmostEqual(stats["mae"], 0.001579759379844714, places=12)
        self.assertAlmostEqual(stats["max_abs_error"], 0.0033907473663514764, places=12)

    def test_loads_current_validation_schema(self):
        payload = {
            "final_score": 0.994,
            "RUS_o": 0.989,
            "representation_metric": {"CKA_f_o": 0.012},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "candidate.json"
            path.write_text(json.dumps(payload))
            metrics = load_validation_metrics(path)
        self.assertEqual(metrics, (0.994, 0.989, 0.012))


if __name__ == "__main__":
    unittest.main()
