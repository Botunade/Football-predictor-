import unittest
from src.models.live_sniper import LiveMomentumSniper


class TestLiveMomentumSniper(unittest.TestCase):
    def test_sniper_evaluation(self):
        sniper = LiveMomentumSniper()
        # 52nd minute, 0-0 score, 2nd Half Over 0.5 odds offered at 1.48 (+12.5% EV)
        alert = sniper.evaluate_synthetic_fixture(
            home_team="Liverpool",
            away_team="Man City",
            minute=52,
            score="0-0",
            live_odds_2h_over05=1.48
        )
        self.assertIsNotNone(alert)
        self.assertEqual(alert.minute, 52)
        self.assertEqual(alert.recommended_market, "2nd Half Over 0.5")
        self.assertGreater(alert.expected_value_pct, 10.0)
        self.assertEqual(alert.urgency, "HIGH")


if __name__ == "__main__":
    unittest.main()
