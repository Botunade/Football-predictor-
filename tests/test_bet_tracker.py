import unittest
from pathlib import Path
import tempfile
from src.automation.bet_tracker import BetTracker


class TestBetTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_bets.db"
        self.tracker = BetTracker(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_and_settle_bet(self):
        legs = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "selection": "1st Half Over 0.5", "odds": 1.42},
            {"home_team": "Real Madrid", "away_team": "Barcelona", "selection": "Over 2.5", "odds": 1.55}
        ]
        bet_id = self.tracker.record_bet(
            booking_code="TEST01",
            stake=100.0,
            combined_odds=2.20,
            legs=legs,
            is_dry_run=False
        )
        self.assertEqual(bet_id, 1)

        bets = self.tracker.get_bets()
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets[0]["booking_code"], "TEST01")
        self.assertEqual(len(bets[0]["legs"]), 2)
        self.assertEqual(bets[0]["status"], "PENDING")

        # Settle bet as WON
        settled = self.tracker.settle_bet(bet_id=1, status="WON")
        self.assertTrue(settled)

        summary = self.tracker.get_pnl_summary()
        self.assertEqual(summary["won_bets"], 1)
        self.assertEqual(summary["total_staked_ngn"], 100.0)
        self.assertEqual(summary["total_payout_ngn"], 220.0)
        self.assertEqual(summary["net_profit_ngn"], 120.0)
        self.assertEqual(summary["roi_pct"], 120.0)
        self.assertEqual(summary["win_rate_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
