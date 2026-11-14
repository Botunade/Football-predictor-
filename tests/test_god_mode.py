"""
Unit tests for God Mode Imminent High-Probability Accumulator Engine.
"""

import time
import unittest
from src.models.acca_simulator import AccaSimulator


class TestGodMode(unittest.TestCase):

    def setUp(self):
        now_ts = int(time.time())
        # Mock fixtures starting in 30 mins, 1 hour, 2 hours, 3 hours
        self.mock_fixtures = [
            {
                "home_team": "Bayern Munich",
                "away_team": "Bochum",
                "league": "Bundesliga",
                "sporty_id": "SP_01",
                "start_time_ts": now_ts + 1800,  # 30 mins
                "match_time": "Today 21:00",
                "odds_home": 1.18,
                "odds_draw": 7.50,
                "odds_away": 15.0,
                "market_odds": {
                    "Over 0.5": 1.05,
                    "Under 4.5": 1.30,
                    "1X": 1.06,
                    "Over 1.5": 1.15
                },
                "sources": ["SportyBet", "Forebet"]
            },
            {
                "home_team": "PSG",
                "away_team": "Metz",
                "league": "Ligue 1",
                "sporty_id": "SP_02",
                "start_time_ts": now_ts + 3600,  # 1 hour
                "match_time": "Today 21:30",
                "odds_home": 1.22,
                "odds_draw": 6.00,
                "odds_away": 12.0,
                "market_odds": {
                    "Over 0.5": 1.06,
                    "Under 4.5": 1.25,
                    "1X": 1.08,
                    "Over 1.5": 1.18
                },
                "sources": ["SportyBet"]
            },
            {
                "home_team": "Real Madrid",
                "away_team": "Leganes",
                "league": "LaLiga",
                "sporty_id": "SP_03",
                "start_time_ts": now_ts + 7200,  # 2 hours
                "match_time": "Today 22:30",
                "odds_home": 1.28,
                "odds_draw": 5.50,
                "odds_away": 10.0,
                "market_odds": {
                    "Over 0.5": 1.07,
                    "Under 4.5": 1.22,
                    "1X": 1.10,
                    "Over 1.5": 1.20
                },
                "sources": ["SportyBet", "PredictZ"]
            }
        ]
        self.sim = AccaSimulator(self.mock_fixtures)

    def test_god_mode_execution(self):
        res = self.sim.simulate_god_mode(max_odds=2.00, min_prob=0.85, max_hours_ahead=4.0, min_legs=2)
        self.assertTrue(res["success"])
        self.assertEqual(res["mode"], "GOD_MODE")
        slip = res["optimal_slip"]
        self.assertLessEqual(slip["combined_odds"], 2.00)
        self.assertGreaterEqual(slip["joint_probability"], 80.0)
        self.assertGreaterEqual(slip["total_legs"], 2)

    def test_god_mode_dossier_content(self):
        res = self.sim.simulate_god_mode(max_odds=2.00, min_prob=0.85, max_hours_ahead=4.0, min_legs=2)
        dossier = res["dossier"]
        self.assertGreaterEqual(len(dossier), 2)
        for d in dossier:
            self.assertIn("final_verdict", d)
            self.assertIn("safety_score", d)
            self.assertIn("lambda_exp", d)
            self.assertIn("time_badge", d)


if __name__ == "__main__":
    unittest.main()
