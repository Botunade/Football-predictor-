"""
Unit tests for V5 Quantitative Intelligence, Bivariate Dixon-Coles,
Probability Calibration, League Tiers, Market EV, and Model Monitoring.
"""

import unittest
from src.models.v5_dixon_coles import DixonColesEngine
from src.models.v5_calibration import get_calibrator
from src.models.v5_league_intelligence import get_league_intelligence
from src.models.v5_market_intelligence import get_market_intelligence
from src.models.v5_ensemble import get_ensemble_engine
from src.models.v5_decision_engine import get_decision_engine
from src.models.v5_monitoring import get_model_monitor


class TestV5Intelligence(unittest.TestCase):

    def setUp(self):
        self.dc = DixonColesEngine()
        self.calibrator = get_calibrator()
        self.league_intel = get_league_intelligence()
        self.market_intel = get_market_intelligence()
        self.ensemble = get_ensemble_engine()
        self.monitor = get_model_monitor()

    def test_dixon_coles_coherence(self):
        lh, la = self.dc.estimate_lambdas_from_odds(1.90, 3.40, 4.00)
        self.assertGreater(lh, 0.5)
        self.assertGreater(la, 0.3)
        markets = self.dc.generate_all_market_probabilities(lh, la)
        # Sum of 1X2 should be close to 1.0
        sum_1x2 = markets["Home"] + markets["Draw"] + markets["Away"]
        self.assertAlmostEqual(sum_1x2, 1.0, delta=0.03)
        self.assertGreater(markets["Over 0.5"], 0.85)

    def test_probability_calibration(self):
        p_cal = self.calibrator.calibrate_binary_prob(0.96)
        # Calibrator dampens overconfidence in sports markets
        self.assertLessEqual(p_cal, 0.96)
        self.assertGreater(p_cal, 0.88)

        p_h, p_d, p_a = self.calibrator.calibrate_1x2_distribution(0.60, 0.25, 0.15)
        self.assertAlmostEqual(p_h + p_d + p_a, 1.0, delta=0.001)

    def test_league_intelligence_tiering(self):
        t_a = self.league_intel.classify_league("English Premier League")
        self.assertEqual(t_a["tier"], "Tier A")
        self.assertTrue(t_a["auto_bet_allowed"])

        t_b = self.league_intel.classify_league("Colombia - Liga DIMAYOR")
        self.assertEqual(t_b["tier"], "Tier B")

        t_c = self.league_intel.classify_league("Panama - Liga Prom Reserves")
        self.assertEqual(t_c["tier"], "Tier C")
        self.assertFalse(t_c["auto_bet_allowed"])

    def test_market_intelligence_and_ev(self):
        fair = self.market_intel.remove_margin_proportional([2.0, 3.4, 3.8])
        self.assertAlmostEqual(sum(fair), 1.0, delta=0.001)

        ev = self.market_intel.compute_expected_value(0.60, 1.85)
        # EV = (0.60 * 1.85) - 1.0 = 0.11 (+11%)
        self.assertAlmostEqual(ev["ev_pct"], 11.0, delta=0.5)
        self.assertTrue(ev["is_positive_ev"])

        clv = self.market_intel.compute_clv(2.10, 1.95)
        self.assertTrue(clv["beat_closing"])
        self.assertGreater(clv["clv_pct"], 0.0)

        kelly = self.market_intel.calculate_quarter_kelly_stake(0.60, 1.85, bankroll=1000.0)
        self.assertTrue(kelly["qualifies_for_wager"])
        self.assertGreater(kelly["recommended_stake_ngn"], 0.0)

    def test_model_monitoring_and_drift(self):
        summary = self.monitor.get_model_intelligence_summary()
        self.assertIn("log_loss", summary)
        self.assertIn("brier_score", summary)
        self.assertIn("expected_calibration_error", summary)
        self.assertIn("mean_clv_pct", summary)
        self.assertIn("calibration_bins", summary)


if __name__ == "__main__":
    unittest.main()
