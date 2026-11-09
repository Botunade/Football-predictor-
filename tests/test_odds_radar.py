import unittest
from pathlib import Path
import tempfile
from src.models.odds_drift_radar import OddsDriftRadar


class TestOddsDriftRadar(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.radar = OddsDriftRadar(store_path=Path(self.temp_file.name))

    def tearDown(self):
        try:
            Path(self.temp_file.name).unlink()
        except Exception:
            pass

    def test_steam_move_detection(self):
        # 1. Opening price
        self.radar.record_snapshot("Chelsea", "Arsenal", oh=2.30, od=3.40, oa=3.10)

        # 2. Heavy sharp syndicate bet on Chelsea pushes odds down from 2.30 to 1.95 (-15.2% steam move)
        sig = self.radar.record_snapshot("Chelsea", "Arsenal", oh=1.95, od=3.50, oa=3.80)

        self.assertEqual(sig.signal_type, "STEAM_BUY_HOME")
        self.assertEqual(sig.steam_target, "Home")
        self.assertLess(sig.drift_h_pct, -10.0)
        self.assertIn("STEAM MOVE", sig.narrative)


if __name__ == "__main__":
    unittest.main()
