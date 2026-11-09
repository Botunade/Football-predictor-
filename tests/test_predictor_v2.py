"""
Tests for Multi-Sport Predictor V4 models (Standard unittest)
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.predictor import MatchPredictor
from src.feature_engineering.feature_builder import build_features, FeatureBuilder


class TestMatchPredictor(unittest.TestCase):
    """Tests for MatchPredictor class"""

    def setUp(self):
        self.predictor = MatchPredictor()
        self.sample_features = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_xG": 2.0,
            "away_xG": 1.5,
            "home_xGA": 1.0,
            "away_xGA": 1.2,
            "home_xGD": 1.0,
            "away_xGD": 0.3,
            "home_ppda": 10.0,
            "away_ppda": 12.0,
            "home_injury": 0.1,
            "away_injury": 0.2,
            "home_fatigue": 0.15,
            "away_fatigue": 0.25,
            "home_away_advantage": 1.1,
            "rest_days_diff": 0,
            "is_derby": 0,
            "odds_home": 2.1,
            "odds_draw": 3.5,
            "odds_away": 4.2
        }

    def test_predictor_initialization(self):
        self.assertIsNotNone(self.predictor)
        self.assertIsNotNone(self.predictor._h2h_model)
        self.assertIsNotNone(self.predictor._scaler)

    def test_predict_match_returns_dict(self):
        result = self.predictor.predict_match(self.sample_features)
        self.assertIsInstance(result, dict)
        self.assertIn("h2h", result)

    def test_predict_match_h2h_structure(self):
        result = self.predictor.predict_match(self.sample_features)
        h2h = result["h2h"]
        required_fields = ["match", "outcome", "value", "betting_code",
                           "model_probability", "implied_probability"]
        for field in required_fields:
            self.assertIn(field, h2h)

    def test_predict_match_outcome_valid(self):
        result = self.predictor.predict_match(self.sample_features)
        h2h = result["h2h"]
        valid_outcomes = ["Home", "Draw", "Away"]
        self.assertIn(h2h["outcome"], valid_outcomes)

    def test_predict_match_probability_range(self):
        result = self.predictor.predict_match(self.sample_features)
        h2h = result["h2h"]
        self.assertTrue(0 <= h2h["model_probability"] <= 1)
        self.assertTrue(0 <= h2h["implied_probability"] <= 1)

    def test_btts_prediction_structure(self):
        result = self.predictor.predict_match(self.sample_features)
        btts = result.get("btts", {})
        self.assertIn("match", btts)
        self.assertEqual(btts.get("outcome"), "BTTS YES")

    def test_over25_prediction_structure(self):
        result = self.predictor.predict_match(self.sample_features)
        over25 = result.get("over_25", {})
        self.assertIn("match", over25)
        self.assertEqual(over25.get("outcome"), "OVER 2.5")

    def test_non_football_prediction(self):
        features_basketball = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "odds_home": 1.90,
            "odds_draw": 0,
            "odds_away": 2.0
        }
        result = self.predictor.predict_match(features_basketball, sport="basketball")
        self.assertIn("h2h", result)


class TestFeatureBuilder(unittest.TestCase):
    """Tests for FeatureBuilder class"""

    def setUp(self):
        self.builder = FeatureBuilder()
        self.sample_match_data = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "odds_home": 2.1,
            "odds_draw": 3.5,
            "odds_away": 4.2,
            "start_time": "2024-01-15T15:00:00Z",
            "sport": "football"
        }

    def test_build_features_basic(self):
        features = asyncio.run(self.builder.build_features(self.sample_match_data))
        self.assertIsNotNone(features)
        self.assertEqual(features.home_team, "Arsenal")
        self.assertEqual(features.away_team, "Chelsea")
        self.assertEqual(features.odds_home, 2.1)
        self.assertEqual(features.odds_draw, 3.5)
        self.assertEqual(features.odds_away, 4.2)


if __name__ == "__main__":
    unittest.main()
