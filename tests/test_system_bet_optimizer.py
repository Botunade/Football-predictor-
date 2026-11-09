import unittest
from src.models.system_bet_optimizer import SystemBetOptimizer


class TestSystemBetOptimizer(unittest.TestCase):
    def setUp(self):
        self.opt = SystemBetOptimizer()
        self.legs = [
            {"home_team": "A", "away_team": "B", "selected_odds": 1.50},
            {"home_team": "C", "away_team": "D", "selected_odds": 1.60},
            {"home_team": "E", "away_team": "F", "selected_odds": 1.40},
            {"home_team": "G", "away_team": "H", "selected_odds": 1.50},
            {"home_team": "I", "away_team": "J", "selected_odds": 1.80}
        ]

    def test_one_cut_coverage(self):
        res = self.opt.calculate_one_cut_coverage(self.legs, stake=100.0, one_cut_discount_pct=0.10)
        self.assertTrue(res["eligible"])
        self.assertEqual(res["total_legs"], 5)
        # Even if the highest odds leg (1.80) cuts the ticket, the remaining 4 legs payout
        self.assertGreater(res["worst_case_one_cut_payout"], 100.0)
        self.assertEqual(len(res["scenarios"]), 5)

    def test_system_combinations(self):
        res = self.opt.generate_system_combinations(self.legs, system_type="4_of_5", total_stake=100.0)
        self.assertEqual(res["total_combos"], 5)  # 5 choose 4 = 5
        self.assertEqual(res["unit_stake"], 20.0)
        self.assertGreater(res["max_payout_if_all_win"], 100.0)


if __name__ == "__main__":
    unittest.main()
