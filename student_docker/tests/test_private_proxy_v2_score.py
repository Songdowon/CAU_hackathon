import unittest
from pathlib import Path

from tools.private_proxy_v2_score import (
    load_recent_calibration,
    temporal_holdout_audit,
)
from tools.private_proxy_score import fit_rus_gap_calibration


class PrivateProxyV2Tests(unittest.TestCase):
    def test_uses_four_recent_pairs_and_keeps_old_soup_as_holdout(self):
        points, holdouts, guardrail, conservative_error = load_recent_calibration()
        self.assertEqual(len(points), 4)
        self.assertEqual({p.family for p in points}, {"uniform", "gate"})
        self.assertEqual([p.name for p in holdouts], ["soup4_relf"])
        self.assertEqual(guardrail, 0.03)
        self.assertGreaterEqual(conservative_error, 0.00339)

    def test_recent_formula_matches_expected_coefficients(self):
        points, _, _, _ = load_recent_calibration()
        model = fit_rus_gap_calibration(points)
        self.assertAlmostEqual(model.center, 0.9885649560448284, places=12)
        self.assertAlmostEqual(model.intercept, -0.002038734429192418, places=12)
        self.assertAlmostEqual(model.slope, -0.7261730279834654, places=12)

    def test_temporal_holdout_error_is_recorded(self):
        points, holdouts, _, _ = load_recent_calibration()
        audit = temporal_holdout_audit(points, holdouts)
        self.assertEqual(audit["n_holdouts"], 1)
        self.assertAlmostEqual(audit["mae"], 0.0006222118472130056, places=12)
        self.assertAlmostEqual(audit["predictions"][0]["predicted"], 0.990962211847213, places=12)


if __name__ == "__main__":
    unittest.main()
