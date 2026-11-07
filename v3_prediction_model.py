"""
Multi-Sport Predictor V3 - Prediction Model

NOTE: This module is now deprecated. Use src.models.predictor instead.
It's kept for backward compatibility with existing code/tests.

V4 Upgrade: Use src.cli for the new CLI interface and src.models for ML predictions
"""

import numpy as np
import os
import pickle
from typing import Dict, Any

# Import from the new modular structure
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    from models.predictor import MatchPredictor, get_predictor
    _predictor = MatchPredictor()

    def predict_match(features: Dict[str, Any], sport: str = "football") -> Dict[str, Any]:
        """
        Predict match outcome (H2H, BTTS, Over 2.5) and calculate value.
        Maintained for backward compatibility.

        Returns prediction with:
        - h2h: Head-to-head prediction
        - btts: Both Teams To Score prediction
        - over_25: Over 2.5 goals prediction
        """
        return _predictor.predict_match(features, sport)

except ImportError:
    # Fallback to original implementation if new modules not available
    import warnings
    warnings.warn(
        "Using legacy prediction model. Consider migrating to src.models.predictor",
        DeprecationWarning
    )

    # Original mock implementation
    class MockModel:
        def predict_proba(self, X):
            batch_size = X.shape[0]
            probs = np.random.dirichlet(np.ones(3), size=batch_size)
            return probs

    class MockScaler:
        def transform(self, X):
            return X

    model = MockModel()
    scaler = MockScaler()

    if os.path.exists("model_v3.pkl"):
        with open("model_v3.pkl", "rb") as f:
            model = pickle.load(f)
    if os.path.exists("scaler_v3.pkl"):
        with open("scaler_v3.pkl", "rb") as f:
            scaler = pickle.load(f)

    def train_model(X_train, y_train):
        """Train model - requires sklearn and xgboost"""
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
        clf.fit(X_train, y_train)

        with open("model_v3.pkl", "wb") as f:
            pickle.dump(clf, f)

        return clf

    def predict_matches(matches_features, sport="football"):
        """Predict multiple matches"""
        return [predict_match(f, sport) for f in matches_features]

    def predict_match(features, sport="football"):
        """
        Predict match outcome (H2H, BTTS, Over 2.5) and calculate value.
        Returns only final predictions with required fields.
        """
        if sport != "football":
            # Simplified prediction for other sports
            model_prob_home = 0.5
            odds_home = features.get("odds_home", 2.0)
            implied_prob_home = 1 / odds_home if odds_home and odds_home > 0 else 0.5
            value_home = model_prob_home - implied_prob_home

            return {
                "h2h": {
                    "match": f"{features['home_team']} vs {features['away_team']}",
                    "outcome": "Home",
                    "value": round(value_home, 3),
                    "betting_code": f"{features['home_team'].replace(' ', '_').upper()}_VS_{features['away_team'].replace(' ', '_').upper()}",
                    "model_probability": round(model_prob_home, 3),
                    "implied_probability": round(implied_prob_home, 3)
                }
            }

        # Feature names for prediction
        feature_names = [
            "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
            "home_ppda", "away_ppda", "home_injury", "away_injury",
            "home_fatigue", "away_fatigue", "home_away_advantage",
            "rest_days_diff", "is_derby"
        ]

        X = np.array([[features.get(f, 0) for f in feature_names]])
        X_scaled = scaler.transform(X)

        # H2H Probabilities
        probs = model.predict_proba(X_scaled)[0]  # [Prob_Home, Prob_Draw, Prob_Away]

        outcomes = ["Home", "Draw", "Away"]
        best_idx = int(np.argmax(probs))
        model_prob_best = probs[best_idx]

        # Value calculation
        odds_map = {0: "odds_home", 1: "odds_draw", 2: "odds_away"}
        target_odds = features.get(odds_map[best_idx], 0)
        implied_prob_best = 1 / target_odds if target_odds and target_odds > 0 else 0
        value_best = model_prob_best - implied_prob_best

        # BTTS prediction
        btts_prob = (
            features.get("home_xG", 1.5) * features.get("away_xGA", 1.5) +
            features.get("away_xG", 1.5) * features.get("home_xGA", 1.5)
        ) / 6.0
        btts_prob = min(max(btts_prob, 0.1), 0.9)

        odds_btts = features.get("odds_btts", 0)
        implied_prob_btts = 1 / odds_btts if odds_btts and odds_btts > 0 else 0
        value_btts = btts_prob - implied_prob_btts if odds_btts else -1

        # Over 2.5 prediction
        over_25_prob = (
            features.get("home_xG", 1.5) + features.get("away_xG", 1.5) +
            features.get("home_xGA", 1.5) + features.get("away_xGA", 1.5)
        ) / 7.0
        over_25_prob = min(max(over_25_prob, 0.1), 0.9)

        odds_over_25 = features.get("odds_over_25", 0)
        implied_prob_over_25 = 1 / odds_over_25 if odds_over_25 and odds_over_25 > 0 else 0
        value_over_25 = over_25_prob - implied_prob_over_25 if odds_over_25 else -1

        match_str = f"{features['home_team']} vs {features['away_team']}"
        bet_code = match_str.replace(' ', '_').upper()

        return {
            "h2h": {
                "match": match_str,
                "outcome": outcomes[best_idx],
                "value": round(value_best, 3),
                "betting_code": bet_code,
                "model_probability": round(model_prob_best, 3),
                "implied_probability": round(implied_prob_best, 3)
            },
            "btts": {
                "match": match_str,
                "outcome": "BTTS YES",
                "value": round(value_btts, 3),
                "betting_code": f"BTTS_{bet_code}",
                "model_probability": round(btts_prob, 3),
                "implied_probability": round(implied_prob_btts, 3)
            },
            "over_25": {
                "match": match_str,
                "outcome": "OVER 2.5",
                "value": round(value_over_25, 3),
                "betting_code": f"O25_{bet_code}",
                "model_probability": round(over_25_prob, 3),
                "implied_probability": round(implied_prob_over_25, 3)
            }
        }


__all__ = ['predict_match', 'predict_matches', 'train_model', 'get_predictor']