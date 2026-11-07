"""
Match Prediction Module
ML-powered prediction with value calculation and confidence scoring
"""

import pickle
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from ..feature_engineering.feature_scaler import FeatureScaler
    from ..data_extractor.scraper_utils import calculate_fair_probabilities, calculate_kelly_stake
except (ImportError, ValueError):
    try:
        from feature_engineering.feature_scaler import FeatureScaler
        from data_extractor.scraper_utils import calculate_fair_probabilities, calculate_kelly_stake
    except ImportError:
        from src.feature_engineering.feature_scaler import FeatureScaler
        from src.data_extractor.scraper_utils import calculate_fair_probabilities, calculate_kelly_stake

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured prediction result"""
    # Head-to-head prediction
    match: str
    outcome: str  # "Home", "Draw", or "Away"
    value: float
    betting_code: str
    model_probability: float
    implied_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "outcome": self.outcome,
            "value": round(self.value, 4),
            "betting_code": self.betting_code,
            "model_probability": round(self.model_probability, 4),
            "implied_probability": round(self.implied_probability, 4)
        }


@dataclass
class MarketPrediction:
    """Prediction for a specific market type"""
    match: str
    outcome: str
    value: float
    betting_code: str
    model_probability: float
    implied_probability: float
    is_value_bet: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "outcome": self.outcome,
            "value": round(self.value, 4),
            "betting_code": self.betting_code,
            "model_probability": round(self.model_probability, 4),
            "implied_probability": round(self.implied_probability, 4),
            "is_value_bet": self.is_value_bet
        }


class MatchPredictor:
    """
    ML-powered match predictor with XGBoost/GBDT models
    Supports H2H, BTTS, and Over/Under 2.5 markets
    """

    def __init__(self, model_path: Optional[str] = None, scaler_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else Path("model_v4.pkl")
        self.scaler_path = Path(scaler_path) if scaler_path else Path("scaler_v4.pkl")

        # Initialize models
        self._h2h_model = None
        self._btts_model = None
        self._over25_model = None
        self._scaler = None

        # Load existing models or create defaults
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load saved models or initialize new ones"""
        try:
            if self.model_path.exists():
                self._load_models()
            else:
                self._initialize_models()
        except Exception as e:
            logger.warning(f"Failed to load models: {e}, initializing fresh models")
            self._initialize_models()

    def _initialize_models(self):
        """Initialize fresh model instances"""
        # H2H model: predicts Home/Draw/Away
        self._h2h_model = CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            cv=3,
            method='sigmoid'
        )

        # BTTS model: binary classifier
        self._btts_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )

        # Over 2.5 model: binary classifier
        self._over25_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )

        # Scaler
        self._scaler = FeatureScaler()

    def _load_models(self):
        """Load models from disk"""
        with open(self.model_path, 'rb') as f:
            models_dict = pickle.load(f)
            self._h2h_model = models_dict.get('h2h_model')
            self._btts_model = models_dict.get('btts_model')
            self._over25_model = models_dict.get('over25_model')

        with open(self.scaler_path, 'rb') as f:
            scaler_dict = pickle.load(f)
            self._scaler = FeatureScaler()
            self._scaler.set_params(scaler_dict)

    def save_models(self):
        """Save models to disk"""
        models_dict = {
            'h2h_model': self._h2h_model,
            'btts_model': self._btts_model,
            'over25_model': self._over25_model,
            'version': 'v4.0',
            'trained_at': datetime.now(timezone.utc).isoformat()
        }

        with open(self.model_path, 'wb') as f:
            pickle.dump(models_dict, f)

        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self._scaler.get_params(), f)

    def fit(self, X: np.ndarray, y_h2h: np.ndarray, y_btts: np.ndarray, y_over25: np.ndarray):
        """
        Train all models

        Args:
            X: Feature matrix
            y_h2h: H2H outcomes (0=Home, 1=Draw, 2=Away)
            y_btts: BTTS outcomes (0=No, 1=Yes)
            y_over25: Over 2.5 outcomes (0=No, 1=Yes)
        """
        logger.info(f"Training models with {X.shape[0]} samples")

        # Fit scaler
        X_scaled = self._scaler.fit_transform(X)

        # Train H2H model
        self._h2h_model.fit(X_scaled, y_h2h)

        # Train BTTS model
        self._btts_model.fit(X_scaled, y_btts)

        # Train Over 2.5 model
        self._over25_model.fit(X_scaled, y_over25)

        logger.info("Model training complete")

    def predict_match(self, features: Dict[str, Any], sport: str = "football") -> Dict[str, Any]:
        """
        Predict match outcome and calculate value

        Args:
            features: Match features dictionary
            sport: Sport type

        Returns:
            Dictionary with H2H, BTTS, and Over 2.5 predictions
        """
        match_str = f"{features['home_team']} vs {features['away_team']}"
        bet_code = match_str.replace(' ', '_').upper()

        # For non-football sports, use simplified prediction
        if sport != "football":
            return self._predict_non_football(features, match_str, bet_code)

        # Prepare feature vector
        feature_names = [
            "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
            "home_ppda", "away_ppda", "home_injury", "away_injury",
            "home_fatigue", "away_fatigue", "home_away_advantage",
            "rest_days_diff", "is_derby"
        ]

        X = np.array([[features.get(f, 0) for f in feature_names]])

        # Scale features
        try:
            X_scaled = self._scaler.transform(X)
        except Exception:
            X_scaled = X  # Use unscaled if scaler not fitted

        # H2H Prediction
        h2h_result = self._predict_h2h(X_scaled[0], features, match_str, bet_code)

        # BTTS Prediction
        btts_result = self._predict_btts(X_scaled[0], features, match_str, bet_code)

        # Over 2.5 Prediction
        over25_result = self._predict_over25(X_scaled[0], features, match_str, bet_code)

        # Full Multi-Market Poisson Options (DC, DNB, Over/Under, BTTS, AH, CS, HT/FT, Exact Score)
        # Resolve dynamic fixture-specific expected goals (xG)
        home_xg = float(features.get("home_xg") or features.get("home_xG") or 0.0)
        away_xg = float(features.get("away_xg") or features.get("away_xG") or 0.0)

        # If xG is missing, 0.0, or flat default (1.5, 1.2), derive it dynamically from match odds
        odds_h = float(features.get("odds_home", 0.0))
        odds_d = float(features.get("odds_draw", 0.0))
        odds_a = float(features.get("odds_away", 0.0))

        if (home_xg == 0.0 or (home_xg == 1.5 and away_xg == 1.2)) and (odds_h > 1.0 and odds_a > 1.0):
            r_h = 1.0 / odds_h
            r_d = 1.0 / (odds_d if odds_d > 1.0 else 3.25)
            r_a = 1.0 / odds_a
            s = r_h + r_d + r_a
            p_h = r_h / s
            p_d = r_d / s
            p_a = r_a / s

            tot_xg = max(1.8, min(3.8, 2.5 + 0.6 * (p_h - p_a)))
            home_xg = round(tot_xg * (p_h + 0.5 * p_d), 2)
            away_xg = round(tot_xg * (p_a + 0.5 * p_d), 2)
        elif home_xg == 0.0:
            # Deterministic variance for matches without odds
            h_seed = sum(ord(c) for c in str(features.get("home_team", "H"))) % 7
            a_seed = sum(ord(c) for c in str(features.get("away_team", "A"))) % 7
            home_xg = round(1.2 + (h_seed * 0.15), 2)
            away_xg = round(0.9 + (a_seed * 0.15), 2)

        multi_markets = self._calculate_multimarket_options(home_xg, away_xg, features)

        return {
            "h2h": h2h_result.to_dict(),
            "btts": btts_result.to_dict(),
            "over_25": over25_result.to_dict(),
            "multi_markets": multi_markets
        }

    def _calculate_multimarket_options(self, home_xg: float, away_xg: float, features: Dict) -> Dict[str, Any]:
        """
        Compute full Poisson distribution over goal matrix to generate all betting options:
        - 1X2 Match Outcome
        - Double Chance (1X, 12, X2)
        - Draw No Bet (Home DNB, Away DNB)
        - Over/Under (1.5, 2.5, 3.5)
        - Both Teams To Score (BTTS Yes, BTTS No)
        - Top 3 Correct Scores
        """
        import math
        MAX_GOALS = 6

        def poisson_pmf(k: int, lamb: float) -> float:
            lamb = max(0.2, min(5.0, lamb))
            return (lamb ** k) * math.exp(-lamb) / math.factorial(k)

        score_matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                score_matrix[i, j] = poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)

        total_p = np.sum(score_matrix)
        if total_p > 0:
            score_matrix /= total_p

        p_home = float(np.sum(np.tril(score_matrix, -1)))
        p_draw = float(np.sum(np.diag(score_matrix)))
        p_away = float(np.sum(np.triu(score_matrix, 1)))

        # Double Chance
        p_1x = min(0.98, p_home + p_draw)
        p_x2 = min(0.98, p_draw + p_away)
        p_12 = min(0.98, p_home + p_away)

        # Draw No Bet
        p_dnb_home = p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0.5
        p_dnb_away = p_away / (p_home + p_away) if (p_home + p_away) > 0 else 0.5

        # Totals
        p_over_05 = float(1.0 - score_matrix[0, 0])
        p_under_05 = max(0.01, 1.0 - p_over_05)

        p_over_15 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 2]))
        p_under_15 = max(0.02, 1.0 - p_over_15)

        p_over_25 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 3]))
        p_under_25 = max(0.02, 1.0 - p_over_25)

        p_over_35 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 4]))
        p_under_35 = max(0.02, 1.0 - p_over_35)

        p_over_45 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 5]))
        p_under_45 = max(0.01, 1.0 - p_over_45)

        # BTTS
        p_btts_yes = float((1.0 - poisson_pmf(0, home_xg)) * (1.0 - poisson_pmf(0, away_xg)))
        p_btts_no = max(0.02, 1.0 - p_btts_yes)

        # Asian Handicap (-0.5, +0.5, -1.5, +1.5)
        p_ah_home_minus_05 = p_home
        p_ah_away_plus_05 = p_x2
        p_ah_away_minus_05 = p_away
        p_ah_home_plus_05 = p_1x
        p_home_win_by_2plus = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i - j >= 2]))
        p_ah_home_minus_15 = max(0.02, p_home_win_by_2plus)
        p_ah_away_plus_15 = max(0.02, 1.0 - p_ah_home_minus_15)

        p_away_win_by_2plus = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if j - i >= 2]))
        p_ah_away_minus_15 = max(0.02, p_away_win_by_2plus)
        p_ah_home_plus_15 = max(0.02, 1.0 - p_ah_away_minus_15)

        # Clean Sheet
        p_home_cs = float(np.sum([score_matrix[i, 0] for i in range(MAX_GOALS)]))
        p_away_cs = float(np.sum([score_matrix[0, j] for j in range(MAX_GOALS)]))

        # HT/FT Estimates
        p_ht_ft_11 = min(0.95, p_home * 0.65)
        p_ht_ft_x1 = min(0.95, p_home * 0.35)
        p_ht_ft_xx = min(0.95, p_draw * 0.60)
        p_ht_ft_22 = min(0.95, p_away * 0.65)
        p_ht_ft_x2 = min(0.95, p_away * 0.35)

        # Top 5 Correct Scores
        scores_list = []
        for i in range(5):
            for j in range(5):
                scores_list.append((f"{i}-{j}", float(score_matrix[i, j])))
        scores_list.sort(key=lambda x: x[1], reverse=True)
        top_scores = [{"score": s[0], "prob": round(s[1], 3), "fair_odds": round(1.0/s[1], 2) if s[1]>0 else 99.0} for s in scores_list[:5]]

        return {
            "match_outcome": {
                "Home": {"prob": round(p_home, 3), "fair_odds": round(1.0/p_home, 2) if p_home>0 else 99.0},
                "Draw": {"prob": round(p_draw, 3), "fair_odds": round(1.0/p_draw, 2) if p_draw>0 else 99.0},
                "Away": {"prob": round(p_away, 3), "fair_odds": round(1.0/p_away, 2) if p_away>0 else 99.0}
            },
            "double_chance": {
                "1X (Home/Draw)": {"prob": round(p_1x, 3), "fair_odds": round(1.0/p_1x, 2) if p_1x>0 else 99.0},
                "X2 (Draw/Away)": {"prob": round(p_x2, 3), "fair_odds": round(1.0/p_x2, 2) if p_x2>0 else 99.0},
                "12 (Home/Away)": {"prob": round(p_12, 3), "fair_odds": round(1.0/p_12, 2) if p_12>0 else 99.0}
            },
            "draw_no_bet": {
                "Home DNB": {"prob": round(p_dnb_home, 3), "fair_odds": round(1.0/p_dnb_home, 2) if p_dnb_home>0 else 99.0},
                "Away DNB": {"prob": round(p_dnb_away, 3), "fair_odds": round(1.0/p_dnb_away, 2) if p_dnb_away>0 else 99.0}
            },
            "totals": {
                "Over 0.5": {"prob": round(p_over_05, 3), "fair_odds": round(1.0/p_over_05, 2) if p_over_05>0 else 99.0},
                "Under 0.5": {"prob": round(p_under_05, 3), "fair_odds": round(1.0/p_under_05, 2) if p_under_05>0 else 99.0},
                "Over 1.5": {"prob": round(p_over_15, 3), "fair_odds": round(1.0/p_over_15, 2) if p_over_15>0 else 99.0},
                "Under 1.5": {"prob": round(p_under_15, 3), "fair_odds": round(1.0/p_under_15, 2) if p_under_15>0 else 99.0},
                "Over 2.5": {"prob": round(p_over_25, 3), "fair_odds": round(1.0/p_over_25, 2) if p_over_25>0 else 99.0},
                "Under 2.5": {"prob": round(p_under_25, 3), "fair_odds": round(1.0/p_under_25, 2) if p_under_25>0 else 99.0},
                "Over 3.5": {"prob": round(p_over_35, 3), "fair_odds": round(1.0/p_over_35, 2) if p_over_35>0 else 99.0},
                "Under 3.5": {"prob": round(p_under_35, 3), "fair_odds": round(1.0/p_under_35, 2) if p_under_35>0 else 99.0},
                "Over 4.5": {"prob": round(p_over_45, 3), "fair_odds": round(1.0/p_over_45, 2) if p_over_45>0 else 99.0},
                "Under 4.5": {"prob": round(p_under_45, 3), "fair_odds": round(1.0/p_under_45, 2) if p_under_45>0 else 99.0}
            },
            "btts": {
                "BTTS Yes": {"prob": round(p_btts_yes, 3), "fair_odds": round(1.0/p_btts_yes, 2) if p_btts_yes>0 else 99.0},
                "BTTS No": {"prob": round(p_btts_no, 3), "fair_odds": round(1.0/p_btts_no, 2) if p_btts_no>0 else 99.0}
            },
            "both_teams_to_score": {
                "BTTS Yes": {"prob": round(p_btts_yes, 3), "fair_odds": round(1.0/p_btts_yes, 2) if p_btts_yes>0 else 99.0},
                "BTTS No": {"prob": round(p_btts_no, 3), "fair_odds": round(1.0/p_btts_no, 2) if p_btts_no>0 else 99.0}
            },
            "asian_handicap": {
                "Home -0.5": {"prob": round(p_ah_home_minus_05, 3), "fair_odds": round(1.0/p_ah_home_minus_05, 2) if p_ah_home_minus_05>0 else 99.0},
                "Home +0.5": {"prob": round(p_ah_home_plus_05, 3), "fair_odds": round(1.0/p_ah_home_plus_05, 2) if p_ah_home_plus_05>0 else 99.0},
                "Away -0.5": {"prob": round(p_ah_away_minus_05, 3), "fair_odds": round(1.0/p_ah_away_minus_05, 2) if p_ah_away_minus_05>0 else 99.0},
                "Away +0.5": {"prob": round(p_ah_away_plus_05, 3), "fair_odds": round(1.0/p_ah_away_plus_05, 2) if p_ah_away_plus_05>0 else 99.0},
                "Home -1.5": {"prob": round(p_ah_home_minus_15, 3), "fair_odds": round(1.0/p_ah_home_minus_15, 2) if p_ah_home_minus_15>0 else 99.0},
                "Away +1.5": {"prob": round(p_ah_away_plus_15, 3), "fair_odds": round(1.0/p_ah_away_plus_15, 2) if p_ah_away_plus_15>0 else 99.0}
            },
            "clean_sheet": {
                "Home Clean Sheet": {"prob": round(p_home_cs, 3), "fair_odds": round(1.0/p_home_cs, 2) if p_home_cs>0 else 99.0},
                "Away Clean Sheet": {"prob": round(p_away_cs, 3), "fair_odds": round(1.0/p_away_cs, 2) if p_away_cs>0 else 99.0}
            },
            "halftime_fulltime": {
                "1/1 (Home/Home)": {"prob": round(p_ht_ft_11, 3), "fair_odds": round(1.0/p_ht_ft_11, 2) if p_ht_ft_11>0 else 99.0},
                "X/1 (Draw/Home)": {"prob": round(p_ht_ft_x1, 3), "fair_odds": round(1.0/p_ht_ft_x1, 2) if p_ht_ft_x1>0 else 99.0},
                "X/X (Draw/Draw)": {"prob": round(p_ht_ft_xx, 3), "fair_odds": round(1.0/p_ht_ft_xx, 2) if p_ht_ft_xx>0 else 99.0},
                "X/2 (Draw/Away)": {"prob": round(p_ht_ft_x2, 3), "fair_odds": round(1.0/p_ht_ft_x2, 2) if p_ht_ft_x2>0 else 99.0},
                "2/2 (Away/Away)": {"prob": round(p_ht_ft_22, 3), "fair_odds": round(1.0/p_ht_ft_22, 2) if p_ht_ft_22>0 else 99.0}
            },
            "top_correct_scores": top_scores
        }

    def _predict_h2h(self, X: np.ndarray, features: Dict, match_str: str, bet_code: str) -> PredictionResult:
        """Predict Head-to-Head outcome"""
        outcomes = ["Home", "Draw", "Away"]

        try:
            # Get calibrated model probabilities
            probs = self._h2h_model.predict_proba(X.reshape(1, -1))[0]

            # Blend with scraped external algorithmic consensus (Forebet / PredictZ) if available
            c_h = features.get("consensus_home_prob")
            c_d = features.get("consensus_draw_prob")
            c_a = features.get("consensus_away_prob")
            if c_h is not None and not (round(c_h, 2) == 0.33 and round(c_d, 2) == 0.33):
                consensus_probs = np.array([c_h, c_d, c_a])
                probs = 0.5 * probs + 0.5 * consensus_probs
                probs = probs / np.sum(probs)

            # Get odds
            odds_h = float(features.get("odds_home", 0.0))
            odds_d = float(features.get("odds_draw", 0.0))
            odds_a = float(features.get("odds_away", 0.0))

            # Remove bookmaker vig to get true market implied probabilities
            if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
                fair_implied_list = calculate_fair_probabilities([odds_h, odds_d, odds_a])
            else:
                fair_implied_list = [1.0/odds_h if odds_h > 0 else 0.33,
                                     1.0/odds_d if odds_d > 0 else 0.33,
                                     1.0/odds_a if odds_a > 0 else 0.34]

            best_idx = int(np.argmax(probs))
            model_prob = float(probs[best_idx])
            implied_prob = float(fair_implied_list[best_idx])
            value = model_prob - implied_prob

            return PredictionResult(
                match=match_str,
                outcome=outcomes[best_idx],
                value=value,
                betting_code=bet_code,
                model_probability=model_prob,
                implied_probability=implied_prob
            )
        except Exception as e:
            logger.error(f"H2H prediction error: {e}")
            # Fallback to simple odds-based
            odds_home = features.get("odds_home", 2.0)
            implied = 1.0 / odds_home if odds_home > 0 else 0.5
            return PredictionResult(
                match=match_str,
                outcome="Home",
                value=implied - implied,
                betting_code=bet_code,
                model_probability=implied,
                implied_probability=implied
            )

    def _predict_btts(self, X: np.ndarray, features: Dict, match_str: str, bet_code: str) -> MarketPrediction:
        """Predict Both Teams to Score"""
        try:
            prob = self._btts_model.predict_proba(X.reshape(1, -1))[0][1]

            # Calculate predicted probability using xG method
            xg_method_prob = (
                features.get("home_xG", 1.5) * features.get("away_xGA", 1.5) +
                features.get("away_xG", 1.5) * features.get("home_xGA", 1.5)
            ) / 6.0
            prob = max(0.1, min(0.9, prob if 0 < prob < 1 else xg_method_prob))

            odds_btts = features.get("odds_btts", 0.0)
            implied = 1.0 / odds_btts if odds_btts > 0 else 0.0
            value = prob - implied

            return MarketPrediction(
                match=match_str,
                outcome="BTTS YES",
                value=value,
                betting_code=f"BTTS_{bet_code}",
                model_probability=prob,
                implied_probability=implied,
                is_value_bet=value > 0.08
            )
        except Exception:
            # Fallback calculation
            prob = 0.5
            return MarketPrediction(
                match=match_str,
                outcome="BTTS YES",
                value=-1.0,
                betting_code=f"BTTS_{bet_code}",
                model_probability=prob,
                implied_probability=0.0,
                is_value_bet=False
            )

    def _predict_over25(self, X: np.ndarray, features: Dict, match_str: str, bet_code: str) -> MarketPrediction:
        """Predict Over 2.5 goals"""
        try:
            prob = self._over25_model.predict_proba(X.reshape(1, -1))[0][1]

            # Calculate using xG method
            xg_method_prob = (
                features.get("home_xG", 1.5) +
                features.get("away_xG", 1.5) +
                features.get("home_xGA", 1.5) +
                features.get("away_xGA", 1.5)
            ) / 7.0
            prob = max(0.1, min(0.9, prob if 0 < prob < 1 else xg_method_prob))

            odds_over = features.get("odds_over_25", 0.0)
            implied = 1.0 / odds_over if odds_over > 0 else 0.0
            value = prob - implied

            return MarketPrediction(
                match=match_str,
                outcome="OVER 2.5",
                value=value,
                betting_code=f"O25_{bet_code}",
                model_probability=prob,
                implied_probability=implied,
                is_value_bet=value > 0.08
            )
        except Exception:
            # Fallback calculation
            prob = 0.5
            return MarketPrediction(
                match=match_str,
                outcome="OVER 2.5",
                value=-1.0,
                betting_code=f"O25_{bet_code}",
                model_probability=prob,
                implied_probability=0.0,
                is_value_bet=False
            )

    def _predict_non_football(self, features: Dict, match_str: str, bet_code: str) -> Dict[str, Any]:
        """Simplified prediction for non-football sports"""
        odds_home = features.get("odds_home", 2.0)
        model_prob = 0.5  # Neutral for non-football
        implied = 1.0 / odds_home if odds_home > 0 else 0.5

        return {
            "h2h": {
                "match": match_str,
                "outcome": "Home",
                "value": model_prob - implied,
                "betting_code": bet_code,
                "model_probability": round(model_prob, 4),
                "implied_probability": round(implied, 4)
            },
            "btts": {},
            "over_25": {}
        }

    def predict_matches(self, features_list: List[Dict], sport: str = "football") -> List[Dict]:
        """Predict multiple matches"""
        return [self.predict_match(f, sport) for f in features_list]


# Default predictor instance
default_predictor = None


def get_predictor(model_path: Optional[str] = None) -> MatchPredictor:
    """Get or create default predictor instance"""
    global default_predictor
    if default_predictor is None:
        default_predictor = MatchPredictor(model_path)
    return default_predictor


def predict_match(features: Dict, sport: str = "football") -> Dict:
    """
    Convenience function for single match prediction
    Maintains backward compatibility with v3_prediction_model
    """
    predictor = get_predictor()
    return predictor.predict_match(features, sport)