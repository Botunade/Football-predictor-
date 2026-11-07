"""
Tests for Multi-Sport Predictor V4 models
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.predictor import MatchPredictor, PredictionResult
from feature_engineering.feature_builder import build_features, FeatureBuilder


class TestMatchPredictor:
    """Tests for MatchPredictor class"""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for testing"""
        return MatchPredictor()

    def test_predictor_initialization(self, predictor):
        """Test predictor initializes correctly"""
        assert predictor is not None
        assert predictor._h2h_model is not None
        assert predictor._scaler is not None

    def test_predict_match_returns_dict(self, predictor, sample_features):
        """Test that predict_match returns proper dictionary"""
        result = predictor.predict_match(sample_features)

        assert isinstance(result, dict)
        assert "h2h" in result

    def test_predict_match_h2h_structure(self, predictor, sample_features):
        """Test H2H prediction has correct structure"""
        result = predictor.predict_match(sample_features)
        h2h = result["h2h"]

        required_fields = ["match", "outcome", "value", "betting_code",
                          "model_probability", "implied_probability"]
        for field in required_fields:
            assert field in h2h, f"Missing field: {field}"

    def test_predict_match_outcome_valid(self, predictor, sample_features):
        """Test outcome is one of valid values"""
        result = predictor.predict_match(sample_features)
        h2h = result["h2h"]

        valid_outcomes = ["Home", "Draw", "Away"]
        assert h2h["outcome"] in valid_outcomes

    def test_predict_match_probability_range(self, predictor, sample_features):
        """Test probabilities are in valid range"""
        result = predictor.predict_match(sample_features)
        h2h = result["h2h"]

        assert 0 <= h2h["model_probability"] <= 1
        assert 0 <= h2h["implied_probability"] <= 1

    def test_predict_match_value_calculation(self, predictor, sample_features):
        """Test value is calculated correctly"""
        result = predictor.predict_match(sample_features)
        h2h = result["h2h"]

        expected_value = h2h["model_probability"] - h2h["implied_probability"]
        assert abs(h2h["value"] - expected_value) < 0.001

    def test_predict_matches_multiple(self, predictor, sample_features):
        """Test prediction of multiple matches"""
        features_list = [sample_features.copy(), sample_features.copy()]
        results = predictor.predict_matches(features_list)

        assert len(results) == 2
        for result in results:
            assert "h2h" in result

    def test_btts_prediction_structure(self, predictor, sample_features):
        """Test BTTS prediction structure"""
        result = predictor.predict_match(sample_features)
        btts = result["btts"]

        assert "match" in btts
        assert "outcome" in btts
        assert btts["outcome"] == "BTTS YES"

    def test_over25_prediction_structure(self, predictor, sample_features):
        """Test Over 2.5 prediction structure"""
        result = predictor.predict_match(sample_features)
        over25 = result["over_25"]

        assert "match" in over25
        assert "outcome" in over25
        assert over25["outcome"] == "OVER 2.5"

    def test_fallback_for_missing_odds(self):
        """Test prediction handles missing odds gracefully"""
        predictor = MatchPredictor()

        features_no_odds = {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_xG": 1.5,
            "away_xG": 1.2,
            "home_xGA": 1.0,
            "away_xGA": 1.1,
            # No odds provided
        }

        result = predictor.predict_match(features_no_odds)
        assert "h2h" in result
        # Should still produce a result with default odds

    def test_non_football_prediction(self):
        """Test simplified prediction for non-football sports"""
        predictor = MatchPredictor()

        features_basketball = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "odds_home": 1.90,
            "odds_draw": 0,
            "odds_away": 2.0
        }

        result = predictor.predict_match(features_basketball, sport="basketball")
        assert "h2h" in result
        # Non-football uses simplified model

    def test_value_bet_detection(self, predictor, sample_features):
        """Test value bet detection logic"""
        # Set odds that create positive value
        features = sample_features.copy()
        features["odds_home"] = 2.5

        result = predictor.predict_match(features)
        h2h = result["h2h"]

        # Value bet when model_prob > implied_prob
        # implied_prob = 1/2.5 = 0.4
        assert "value" in h2h


class TestFeatureBuilder:
    """Tests for FeatureBuilder class"""

    @pytest.fixture
    def builder(self):
        return FeatureBuilder()

    @pytest.mark.asyncio
    async def test_build_features_basic(self, builder, sample_match_data):
        """Test basic feature building"""
        features = await builder.build_features(sample_match_data)

        assert features is not None
        assert hasattr(features, 'home_team')
        assert hasattr(features, 'away_team')
        assert features.home_team == "Arsenal"
        assert features.away_team == "Chelsea"

    @pytest.mark.asyncio
    async def test_build_features_with_odds(self, builder, sample_match_data):
        """Test feature building preserves odds"""
        features = await builder.build_features(sample_match_data)

        assert features.odds_home == 2.1
        assert features.odds_draw == 3.5
        assert features.odds_away == 4.2

    @pytest.mark.asyncio
    async def test_build_features_to_dict(self, builder, sample_match_data):
        """Test converting features to dictionary"""
        features = await builder.build_features(sample_match_data)
        feature_dict = features.to_dict()

        assert isinstance(feature_dict, dict)
        assert "home_team" in feature_dict
        assert "away_team" in feature_dict

    @pytest.mark.asyncio
    async def test_build_features_to_list(self, builder, sample_match_data):
        """Test converting features to list for model input"""
        features = await builder.build_features(sample_match_data)
        feature_list = features.to_list()

        assert isinstance(feature_list, list)
        assert len(feature_list) == 15  # Default feature count


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_prediction_pipeline(self, sample_match_data):
        """Test complete pipeline from raw data to prediction"""
        # Build features
        features = await build_features(sample_match_data)

        # Predict
        predictor = MatchPredictor()
        prediction = predictor.predict_match(features.to_dict())

        # Validate
        assert "h2h" in prediction
        assert prediction["h2h"]["match"] == "Arsenal vs Chelsea"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])