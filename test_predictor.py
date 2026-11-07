"""
Tests for Multi-Sport Predictor V4 - Updated for modular architecture
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import from new modular structure
from src.models.predictor import predict_match, get_predictor
from src.feature_engineering.feature_builder import build_features, FeatureBuilder
from src.feature_engineering.feature_scaler import FeatureScaler


def async_test(f):
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


class TestFootballPredictor(unittest.TestCase):
    """Tests for the updated predictor"""

    def setUp(self):
        """Create a trained model for testing"""
        # Create a trained predictor
        self.predictor = get_predictor()

    def test_compute_context(self):
        """Test context computation"""
        from src.feature_engineering.feature_builder import FeatureBuilder
        builder = FeatureBuilder()

        fixture = {"fixture": {"id": 123}}
        context = builder.compute_context(fixture)
        self.assertIn("home_away_advantage", context)
        self.assertIn("is_derby", context)

    def test_predict_match_basic(self):
        """Test basic prediction functionality"""
        features = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_xG": 2.0,
            "away_xG": 1.0,
            "odds_home": 2.0,
            "odds_draw": 3.5,
            "odds_away": 4.0,
            "home_xGA": 1.0,
            "away_xGA": 1.2,
            "home_ppda": 10.0,
            "away_ppda": 12.0,
            "home_injury": 0.1,
            "away_injury": 0.2,
            "home_fatigue": 0.15,
            "away_fatigue": 0.25,
            "rest_days_diff": 0,
            "is_derby": 0,
            "home_away_advantage": 1.1
        }

        prediction = predict_match(features)

        self.assertIn("h2h", prediction)
        self.assertIn("btts", prediction)
        self.assertIn("over_25", prediction)

    def test_predict_match_structure(self):
        """Test that prediction returns correct structure"""
        features = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "home_xG": 2.0,
            "away_xG": 1.5,
            "odds_home": 2.1,
            "odds_draw": 3.5,
            "odds_away": 4.0,
            "home_xGA": 1.0,
            "away_xGA": 1.2,
            "home_ppda": 10.0,
            "away_ppda": 12.0,
            "home_injury": 0.1,
            "away_injury": 0.2,
            "home_fatigue": 0.15,
            "away_fatigue": 0.25,
            "rest_days_diff": 0,
            "is_derby": 0,
            "home_away_advantage": 1.1
        }

        prediction = predict_match(features)

        # Check H2H structure
        h2h = prediction["h2h"]
        self.assertIn("match", h2h)
        self.assertIn("outcome", h2h)
        self.assertIn("value", h2h)
        self.assertIn("betting_code", h2h)
        self.assertIn("model_probability", h2h)
        self.assertIn("implied_probability", h2h)

        # Check market is in valid outcomes
        self.assertIn(h2h["outcome"], ["Home", "Draw", "Away"])

        # Check probabilities are in valid range
        self.assertGreaterEqual(h2h["model_probability"], 0)
        self.assertLessEqual(h2h["model_probability"], 1)

    def test_predict_match_value_calculation(self):
        """Test that value calculation is correct"""
        features = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_xG": 2.0,
            "away_xG": 1.5,
            "odds_home": 2.0,
            "odds_draw": 3.5,
            "odds_away": 4.0,
            "home_xGA": 1.0,
            "away_xGA": 1.2,
            "home_ppda": 10.0,
            "away_ppda": 12.0,
            "home_injury": 0.1,
            "away_injury": 0.2,
            "home_fatigue": 0.15,
            "away_fatigue": 0.25,
            "rest_days_diff": 0,
            "is_derby": 0,
            "home_away_advantage": 1.1
        }

        prediction = predict_match(features)

        # Value should be model_prob - implied_prob
        model_prob = prediction["h2h"]["model_probability"]
        implied_prob = prediction["h2h"]["implied_probability"]
        value = prediction["h2h"]["value"]

        self.assertAlmostEqual(value, model_prob - implied_prob, places=3)

    def test_scaler_basic(self):
        """Test feature scaler"""
        import numpy as np

        scaler = FeatureScaler()

        # Use normalized data where expected output is obvious
        X = np.array([[0.0, 0.0], [2.0, 2.0]])
        scaler.fit(X)

        X_scaled = scaler.transform(np.array([[1.0, 1.0], [2.0, 2.0]]))

        # First row should be all 0s (mean)
        self.assertAlmostEqual(X_scaled[0][0], 0.0, places=5)
        self.assertAlmostEqual(X_scaled[0][1], 0.0, places=5)

        # Second row should be (2-1)/1 = 1
        self.assertAlmostEqual(X_scaled[1][0], 1.0, places=5)


class TestFeatureBuilder(unittest.TestCase):
    """Tests for feature engineering"""

    @async_test
    async def test_build_features_basic(self):
        """Test basic feature building"""
        fixture = {
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "odds_home": 2.1,
            "odds_draw": 3.5,
            "odds_away": 4.0
        }

        features = await build_features(fixture, sport="football")

        self.assertEqual(features.home_team, "Arsenal")
        self.assertEqual(features.away_team, "Chelsea")
        self.assertEqual(features.odds_home, 2.1)

    @async_test
    async def test_build_features_with_teams_structure(self):
        """Test feature building with full API structure"""
        fixture = {
            "fixture": {"id": 123},
            "league": {"id": 39, "season": 2024},
            "teams": {
                "home": {"id": 10, "name": "Arsenal"},
                "away": {"id": 20, "name": "Chelsea"}
            },
            "odds_home": 2.0,
            "odds_draw": 3.5,
            "odds_away": 4.0
        }

        features = await build_features(fixture, sport="football")

        self.assertEqual(features.home_team, "Arsenal")
        self.assertEqual(features.away_team, "Chelsea")


if __name__ == '__main__':
    unittest.main()