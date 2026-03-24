import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from full_data_pipeline import build_dataset, compute_context
from v3_prediction_model import predict_match

class TestFootballPredictor(unittest.TestCase):

    def test_compute_context(self):
        fixture = {"fixture": {"id": 123}}
        context = compute_context(fixture)
        self.assertIn("home_away_advantage", context)
        self.assertIn("is_derby", context)

    def test_predict_match(self):
        features = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_xG": 2.0,
            "away_xG": 1.0,
            "odds_home": 2.0
        }
        prediction = predict_match(features)
        self.assertIn("h2h", prediction)
        self.assertIn("btts", prediction)
        self.assertIn("over_25", prediction)
        # 1/2.0 = 0.5 implied prob
        self.assertEqual(prediction["h2h"]["implied_probability"], 0.5)

    @patch('full_data_pipeline.scrape_understat_team')
    @patch('full_data_pipeline.fetch_player_info')
    def test_build_dataset(self, mock_player, mock_understat):
        mock_understat.return_value = {
            "xG": 1.5, "xGA": 1.2, "xGD": 0.3, "NPxG": 1.4, "PPDA": 10.5, "possession": 52.0
        }
        mock_player.return_value = {
            "injury_index": 0.1, "fatigue_index": 0.2, "key_player_dependency": 0.5
        }

        fixtures = [{
            "fixture": {"id": 1},
            "teams": {
                "home": {"id": 10, "name": "Team A"},
                "away": {"id": 20, "name": "Team B"}
            },
            "league": {"id": 39, "season": 2024}
        }]

        df = build_dataset(fixtures)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["home_team"], "Team A")

if __name__ == '__main__':
    unittest.main()
