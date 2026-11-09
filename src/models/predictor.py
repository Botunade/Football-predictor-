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
        # Resolve dynamic fixture-specific expected goals (xG) with Venue-Split Calibration
        home_xg = float(features.get("home_xg") or features.get("home_xG") or 0.0)
        away_xg = float(features.get("away_xg") or features.get("away_xG") or 0.0)

        # If xG is missing, 0.0, or flat default (1.5, 1.2), derive it dynamically from match odds with home advantage
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

            # Expected total goals dynamically scaled by match pace and team balance
            tot_xg = max(1.75, min(3.85, 2.45 + 0.55 * (p_h - p_a)))
            
            # Apply venue-split weight: home teams convert higher share of possession at home (~1.12x factor)
            home_venue_factor = 1.12
            away_venue_factor = 0.88
            raw_h_xg = tot_xg * (p_h + 0.5 * p_d) * home_venue_factor
            raw_a_xg = tot_xg * (p_a + 0.5 * p_d) * away_venue_factor
            
            # Rebalance to expected total goals
            scale = tot_xg / (raw_h_xg + raw_a_xg) if (raw_h_xg + raw_a_xg) > 0 else 1.0
            home_xg = round(raw_h_xg * scale, 2)
            away_xg = round(raw_a_xg * scale, 2)
        elif home_xg == 0.0:
            # Deterministic variance for matches without odds
            h_seed = sum(ord(c) for c in str(features.get("home_team", "H"))) % 7
            a_seed = sum(ord(c) for c in str(features.get("away_team", "A"))) % 7
            home_xg = round(1.25 + (h_seed * 0.14), 2)
            away_xg = round(0.85 + (a_seed * 0.14), 2)

        multi_markets = self._calculate_multimarket_options(home_xg, away_xg, features)

        return {
            "h2h": h2h_result.to_dict(),
            "btts": btts_result.to_dict(),
            "over_25": over25_result.to_dict(),
            "multi_markets": multi_markets
        }

    def _calculate_multimarket_options(self, home_xg: float, away_xg: float, features: Dict) -> Dict[str, Any]:
        """
        Compute Dixon-Coles Bivariate Poisson distribution over score matrix to generate all betting options:
        - Dixon-Coles parameter tau(x, y) with rho = -0.11 corrects the independence flaw in low scores (0-0, 1-1, 1-0, 0-1)
        - 1X2 Match Outcome
        - Double Chance (1X, 12, X2)
        - Draw No Bet (Home DNB, Away DNB)
        - Over/Under (0.5 to 4.5)
        - Both Teams To Score (BTTS Yes, BTTS No)
        - Asian Handicap (-0.5, +0.5, -1.5, +1.5)
        - Clean Sheets
        - Top 5 Correct Scores
        """
        import math
        MAX_GOALS = 6

        def poisson_pmf(k: int, lamb: float) -> float:
            lamb = max(0.2, min(5.0, lamb))
            return (lamb ** k) * math.exp(-lamb) / math.factorial(k)

        def dixon_coles_tau(x: int, y: int, lambda_param: float, mu_param: float, rho: float = -0.11) -> float:
            """
            Dixon-Coles adjustment parameter tau(x, y).
            Corrects low-scoring correlation in football (empirically rho ~ -0.11 for professional leagues).
            Inflates 0-0 and 1-1 probabilities while deflating 1-0 and 0-1.
            """
            if x == 0 and y == 0:
                return 1.0 - (lambda_param * mu_param * rho)
            elif x == 0 and y == 1:
                return 1.0 + (mu_param * rho)
            elif x == 1 and y == 0:
                return 1.0 + (lambda_param * rho)
            elif x == 1 and y == 1:
                return 1.0 - rho
            else:
                return 1.0

        score_matrix = np.zeros((MAX_GOALS, MAX_GOALS))
        for i in range(MAX_GOALS):
            for j in range(MAX_GOALS):
                base_prob = poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)
                tau = dixon_coles_tau(i, j, home_xg, away_xg, rho=-0.11)
                score_matrix[i, j] = max(0.00001, base_prob * tau)

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

        # Totals (Calculated from joint probability matrix)
        p_over_05 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 1]))
        p_under_05 = max(0.01, 1.0 - p_over_05)

        p_over_15 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 2]))
        p_under_15 = max(0.02, 1.0 - p_over_15)

        p_over_25 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 3]))
        p_under_25 = max(0.02, 1.0 - p_over_25)

        p_over_35 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 4]))
        p_under_35 = max(0.02, 1.0 - p_over_35)

        p_over_45 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if i + j >= 5]))
        p_under_45 = max(0.01, 1.0 - p_over_45)

        # BTTS (Exact joint probability where both i >= 1 and j >= 1)
        p_btts_yes = float(np.sum(score_matrix[1:, 1:]))
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

        # Clean Sheet & Team Totals (Full Time)
        p_home_cs = float(np.sum(score_matrix[:, 0]))
        p_away_cs = float(np.sum(score_matrix[0, :]))
        p_home_cs_no = max(0.02, 1.0 - p_home_cs)
        p_away_cs_no = max(0.02, 1.0 - p_away_cs)

        p_home_over_05 = max(0.02, 1.0 - p_away_cs)
        p_home_under_05 = max(0.02, p_away_cs)
        p_home_over_15 = float(np.sum(score_matrix[2:, :]))
        p_home_under_15 = max(0.02, 1.0 - p_home_over_15)

        p_away_over_05 = max(0.02, 1.0 - p_home_cs)
        p_away_under_05 = max(0.02, p_home_cs)
        p_away_over_15 = float(np.sum(score_matrix[:, 2:]))
        p_away_under_15 = max(0.02, 1.0 - p_away_over_15)

        # Odd / Even & Multi-Goals
        p_odd_goals = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if (i + j) % 2 == 1]))
        p_even_goals = max(0.02, 1.0 - p_odd_goals)
        p_mg_1_2 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if 1 <= (i + j) <= 2]))
        p_mg_1_3 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if 1 <= (i + j) <= 3]))
        p_mg_2_3 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if 2 <= (i + j) <= 3]))
        p_mg_1_4 = float(np.sum([score_matrix[i, j] for i in range(MAX_GOALS) for j in range(MAX_GOALS) if 1 <= (i + j) <= 4]))

        # Early Payout 1UP / 2UP Probabilities
        p_home_1up = min(0.95, p_home + (p_draw * 0.34) + (p_away * 0.12))
        p_away_1up = min(0.95, p_away + (p_draw * 0.32) + (p_home * 0.10))
        p_home_2up = min(0.94, p_home + (p_draw * 0.08))
        p_away_2up = min(0.94, p_away + (p_draw * 0.07))

        # =====================================================================
        # 1st Half (~44% of xG) & 2nd Half (~56% of xG) Bivariate Matrices
        # =====================================================================
        def build_half_probs(h_scale: float) -> Dict[str, float]:
            hl_h = max(0.12, home_xg * h_scale)
            hl_a = max(0.10, away_xg * h_scale)
            hm = np.zeros((5, 5))
            for i in range(5):
                for j in range(5):
                    bp = poisson_pmf(i, hl_h) * poisson_pmf(j, hl_a)
                    hm[i, j] = max(0.00001, bp * dixon_coles_tau(i, j, hl_h, hl_a, rho=-0.08))
            hm /= np.sum(hm)
            ph = float(np.sum(np.tril(hm, -1)))
            pd = float(np.sum(np.diag(hm)))
            pa = float(np.sum(np.triu(hm, 1)))
            po05 = float(np.sum([hm[i, j] for i in range(5) for j in range(5) if i + j >= 1]))
            po15 = float(np.sum([hm[i, j] for i in range(5) for j in range(5) if i + j >= 2]))
            po25 = float(np.sum([hm[i, j] for i in range(5) for j in range(5) if i + j >= 3]))
            pgg = float(np.sum(hm[1:, 1:]))
            return {
                "home": ph, "draw": pd, "away": pa,
                "1x": min(0.98, ph + pd), "x2": min(0.98, pd + pa), "12": min(0.98, ph + pa),
                "dnb_h": ph / (ph + pa) if (ph + pa) > 0 else 0.5,
                "dnb_a": pa / (ph + pa) if (ph + pa) > 0 else 0.5,
                "o05": po05, "u05": max(0.02, 1.0 - po05),
                "o15": po15, "u15": max(0.02, 1.0 - po15),
                "o25": po25, "u25": max(0.01, 1.0 - po25),
                "gg": pgg, "ng": max(0.02, 1.0 - pgg)
            }

        ht_p = build_half_probs(0.44)
        sh_p = build_half_probs(0.56)

        # Win Either Half & Both Halves O/U 1.5
        p_home_weh = min(0.96, 1.0 - ((1.0 - ht_p["home"]) * (1.0 - sh_p["home"])))
        p_away_weh = min(0.96, 1.0 - ((1.0 - ht_p["away"]) * (1.0 - sh_p["away"])))
        p_bh_over_15 = max(0.02, ht_p["o15"] * sh_p["o15"])
        p_bh_under_15 = max(0.02, ht_p["u15"] * sh_p["u15"])

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

        # Resolve real SportyBet multi-market bookmaker odds if available
        sb_market_odds: Dict[str, float] = {}
        if isinstance(features.get("market_odds"), dict):
            sb_market_odds.update(features["market_odds"])
        try:
            try:
                from data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
            except ImportError:
                from src.data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
            agg = get_stats_aggregator()
            sb_m = agg.find_sportybet_match(str(features.get("home_team", "")), str(features.get("away_team", "")))
            if sb_m and getattr(sb_m, "market_odds", None):
                for mk, mv in sb_m.market_odds.items():
                    if mk not in sb_market_odds or sb_market_odds[mk] <= 1.0:
                        sb_market_odds[mk] = mv
        except Exception:
            pass

        oh_book = float(features.get("odds_home", 0.0) or 0.0)
        od_book = float(features.get("odds_draw", 0.0) or 0.0)
        oa_book = float(features.get("odds_away", 0.0) or 0.0)

        def get_book_odds(key: str, prob_val: float, default_override: float = 0.0) -> float:
            if key in sb_market_odds and sb_market_odds[key] > 1.01:
                return round(float(sb_market_odds[key]), 2)
            if default_override > 1.01:
                return round(default_override, 2)
            p_clamped = max(0.04, min(0.93, prob_val))
            return round(max(1.06, 0.93 / p_clamped), 2)

        raw_bookable = [
            # Full Time 1X2 & Early Payout
            ("Home", "1X2: Home (1)", "Full Time 1X2 & Early Payout", p_home, get_book_odds("Home", p_home, oh_book), False),
            ("Draw", "1X2: Draw (X)", "Full Time 1X2 & Early Payout", p_draw, get_book_odds("Draw", p_draw, od_book), False),
            ("Away", "1X2: Away (2)", "Full Time 1X2 & Early Payout", p_away, get_book_odds("Away", p_away, oa_book), False),
            ("Home (1UP)", "1X2 - 1UP: Home (Early Payout)", "Full Time 1X2 & Early Payout", p_home_1up, get_book_odds("Home (1UP)", p_home_1up), True),
            ("Away (1UP)", "1X2 - 1UP: Away (Early Payout)", "Full Time 1X2 & Early Payout", p_away_1up, get_book_odds("Away (1UP)", p_away_1up), True),
            ("Home (2UP)", "1X2 - 2UP: Home (2-Goal Payout)", "Full Time 1X2 & Early Payout", p_home_2up, get_book_odds("Home (2UP)", p_home_2up), False),
            ("Away (2UP)", "1X2 - 2UP: Away (2-Goal Payout)", "Full Time 1X2 & Early Payout", p_away_2up, get_book_odds("Away (2UP)", p_away_2up), False),

            # Double Chance & Draw No Bet
            ("1X", "Double Chance: 1X (Home/Draw)", "Double Chance & DNB", p_1x, get_book_odds("1X", p_1x), True),
            ("X2", "Double Chance: X2 (Away/Draw)", "Double Chance & DNB", p_x2, get_book_odds("X2", p_x2), True),
            ("12", "Double Chance: 12 (Home/Away)", "Double Chance & DNB", p_12, get_book_odds("12", p_12), True),
            ("Home DNB", "Draw No Bet: Home DNB", "Double Chance & DNB", p_dnb_home, get_book_odds("Home DNB", p_dnb_home), True),
            ("Away DNB", "Draw No Bet: Away DNB", "Double Chance & DNB", p_dnb_away, get_book_odds("Away DNB", p_dnb_away), True),

            # Full Time Goals O/U
            ("Over 0.5", "Full Time: Over 0.5 Goals", "Full Time Goals O/U", p_over_05, get_book_odds("Over 0.5", p_over_05), True),
            ("Over 1.5", "Full Time: Over 1.5 Goals", "Full Time Goals O/U", p_over_15, get_book_odds("Over 1.5", p_over_15), True),
            ("Under 1.5", "Full Time: Under 1.5 Goals", "Full Time Goals O/U", p_under_15, get_book_odds("Under 1.5", p_under_15), False),
            ("Over 2.5", "Full Time: Over 2.5 Goals", "Full Time Goals O/U", p_over_25, get_book_odds("Over 2.5", p_over_25), False),
            ("Under 2.5", "Full Time: Under 2.5 Goals", "Full Time Goals O/U", p_under_25, get_book_odds("Under 2.5", p_under_25), False),
            ("Over 3.5", "Full Time: Over 3.5 Goals", "Full Time Goals O/U", p_over_35, get_book_odds("Over 3.5", p_over_35), False),
            ("Under 3.5", "Full Time: Under 3.5 Goals", "Full Time Goals O/U", p_under_35, get_book_odds("Under 3.5", p_under_35), True),
            ("Under 4.5", "Full Time: Under 4.5 Goals", "Full Time Goals O/U", p_under_45, get_book_odds("Under 4.5", p_under_45), True),

            # 1st Half Markets
            ("1st Half Over 0.5", "1st Half: Over 0.5 Goals", "1st Half Markets", ht_p["o05"], get_book_odds("1st Half Over 0.5", ht_p["o05"]), True),
            ("1st Half Under 0.5", "1st Half: Under 0.5 Goals", "1st Half Markets", ht_p["u05"], get_book_odds("1st Half Under 0.5", ht_p["u05"]), False),
            ("1st Half Over 1.5", "1st Half: Over 1.5 Goals", "1st Half Markets", ht_p["o15"], get_book_odds("1st Half Over 1.5", ht_p["o15"]), False),
            ("1st Half Under 1.5", "1st Half: Under 1.5 Goals", "1st Half Markets", ht_p["u15"], get_book_odds("1st Half Under 1.5", ht_p["u15"]), True),
            ("1st Half Under 2.5", "1st Half: Under 2.5 Goals", "1st Half Markets", ht_p["u25"], get_book_odds("1st Half Under 2.5", ht_p["u25"]), True),
            ("1st Half Home", "1st Half 1X2: Home", "1st Half Markets", ht_p["home"], get_book_odds("1st Half Home", ht_p["home"]), False),
            ("1st Half Draw", "1st Half 1X2: Draw", "1st Half Markets", ht_p["draw"], get_book_odds("1st Half Draw", ht_p["draw"]), False),
            ("1st Half Away", "1st Half 1X2: Away", "1st Half Markets", ht_p["away"], get_book_odds("1st Half Away", ht_p["away"]), False),
            ("1st Half 1X", "1st Half DC: 1X (Home/Draw)", "1st Half Markets", ht_p["1x"], get_book_odds("1st Half 1X", ht_p["1x"]), True),
            ("1st Half X2", "1st Half DC: X2 (Away/Draw)", "1st Half Markets", ht_p["x2"], get_book_odds("1st Half X2", ht_p["x2"]), True),
            ("1st Half 12", "1st Half DC: 12 (Home/Away)", "1st Half Markets", ht_p["12"], get_book_odds("1st Half 12", ht_p["12"]), False),
            ("1st Half DNB Home", "1st Half DNB: Home", "1st Half Markets", ht_p["dnb_h"], get_book_odds("1st Half DNB Home", ht_p["dnb_h"]), True),
            ("1st Half DNB Away", "1st Half DNB: Away", "1st Half Markets", ht_p["dnb_a"], get_book_odds("1st Half DNB Away", ht_p["dnb_a"]), True),
            ("1st Half GG", "1st Half BTTS: Yes (GG)", "1st Half Markets", ht_p["gg"], get_book_odds("1st Half GG", ht_p["gg"]), False),
            ("1st Half NG", "1st Half BTTS: No (NG)", "1st Half Markets", ht_p["ng"], get_book_odds("1st Half NG", ht_p["ng"]), True),

            # 2nd Half Markets
            ("2nd Half Over 0.5", "2nd Half: Over 0.5 Goals", "2nd Half Markets", sh_p["o05"], get_book_odds("2nd Half Over 0.5", sh_p["o05"]), True),
            ("2nd Half Under 0.5", "2nd Half: Under 0.5 Goals", "2nd Half Markets", sh_p["u05"], get_book_odds("2nd Half Under 0.5", sh_p["u05"]), False),
            ("2nd Half Over 1.5", "2nd Half: Over 1.5 Goals", "2nd Half Markets", sh_p["o15"], get_book_odds("2nd Half Over 1.5", sh_p["o15"]), False),
            ("2nd Half Under 1.5", "2nd Half: Under 1.5 Goals", "2nd Half Markets", sh_p["u15"], get_book_odds("2nd Half Under 1.5", sh_p["u15"]), True),
            ("2nd Half Over 2.5", "2nd Half: Over 2.5 Goals", "2nd Half Markets", sh_p["o25"], get_book_odds("2nd Half Over 2.5", sh_p["o25"]), False),
            ("2nd Half Under 2.5", "2nd Half: Under 2.5 Goals", "2nd Half Markets", sh_p["u25"], get_book_odds("2nd Half Under 2.5", sh_p["u25"]), True),
            ("2nd Half Home", "2nd Half 1X2: Home", "2nd Half Markets", sh_p["home"], get_book_odds("2nd Half Home", sh_p["home"]), False),
            ("2nd Half Draw", "2nd Half 1X2: Draw", "2nd Half Markets", sh_p["draw"], get_book_odds("2nd Half Draw", sh_p["draw"]), False),
            ("2nd Half Away", "2nd Half 1X2: Away", "2nd Half Markets", sh_p["away"], get_book_odds("2nd Half Away", sh_p["away"]), False),
            ("2nd Half 1X", "2nd Half DC: 1X (Home/Draw)", "2nd Half Markets", sh_p["1x"], get_book_odds("2nd Half 1X", sh_p["1x"]), True),
            ("2nd Half X2", "2nd Half DC: X2 (Away/Draw)", "2nd Half Markets", sh_p["x2"], get_book_odds("2nd Half X2", sh_p["x2"]), True),
            ("2nd Half 12", "2nd Half DC: 12 (Home/Away)", "2nd Half Markets", sh_p["12"], get_book_odds("2nd Half 12", sh_p["12"]), False),
            ("2nd Half DNB Home", "2nd Half DNB: Home", "2nd Half Markets", sh_p["dnb_h"], get_book_odds("2nd Half DNB Home", sh_p["dnb_h"]), True),
            ("2nd Half DNB Away", "2nd Half DNB: Away", "2nd Half Markets", sh_p["dnb_a"], get_book_odds("2nd Half DNB Away", sh_p["dnb_a"]), True),
            ("2nd Half GG", "2nd Half BTTS: Yes (GG)", "2nd Half Markets", sh_p["gg"], get_book_odds("2nd Half GG", sh_p["gg"]), False),
            ("2nd Half NG", "2nd Half BTTS: No (NG)", "2nd Half Markets", sh_p["ng"], get_book_odds("2nd Half NG", sh_p["ng"]), True),

            # BTTS & Goal Ranges
            ("BTTS Yes", "Both Teams To Score: Yes (GG)", "BTTS & Special Markets", p_btts_yes, get_book_odds("BTTS Yes", p_btts_yes), False),
            ("BTTS No", "Both Teams To Score: No (NG)", "BTTS & Special Markets", p_btts_no, get_book_odds("BTTS No", p_btts_no), False),
            ("Multi-Goals 1-3", "Multi-Goals: 1-3 Goals", "BTTS & Special Markets", p_mg_1_3, get_book_odds("Multi-Goals 1-3", p_mg_1_3), True),
            ("Multi-Goals 2-3", "Multi-Goals: 2-3 Goals", "BTTS & Special Markets", p_mg_2_3, get_book_odds("Multi-Goals 2-3", p_mg_2_3), False),
            ("Multi-Goals 1-4", "Multi-Goals: 1-4 Goals", "BTTS & Special Markets", p_mg_1_4, get_book_odds("Multi-Goals 1-4", p_mg_1_4), True),
            ("Odd Goals", "Odd/Even: Odd Goals", "BTTS & Special Markets", p_odd_goals, get_book_odds("Odd Goals", p_odd_goals), False),
            ("Even Goals", "Odd/Even: Even Goals", "BTTS & Special Markets", p_even_goals, get_book_odds("Even Goals", p_even_goals), False),

            # Team Goals, Clean Sheets & Halves
            ("Home Team Over 0.5", "Home Team: Over 0.5 Goals", "Team Goals & Clean Sheets", p_home_over_05, get_book_odds("Home Team Over 0.5", p_home_over_05), True),
            ("Home Team Over 1.5", "Home Team: Over 1.5 Goals", "Team Goals & Clean Sheets", p_home_over_15, get_book_odds("Home Team Over 1.5", p_home_over_15), False),
            ("Home Team Under 1.5", "Home Team: Under 1.5 Goals", "Team Goals & Clean Sheets", p_home_under_15, get_book_odds("Home Team Under 1.5", p_home_under_15), True),
            ("Away Team Over 0.5", "Away Team: Over 0.5 Goals", "Team Goals & Clean Sheets", p_away_over_05, get_book_odds("Away Team Over 0.5", p_away_over_05), True),
            ("Away Team Over 1.5", "Away Team: Over 1.5 Goals", "Team Goals & Clean Sheets", p_away_over_15, get_book_odds("Away Team Over 1.5", p_away_over_15), False),
            ("Away Team Under 1.5", "Away Team: Under 1.5 Goals", "Team Goals & Clean Sheets", p_away_under_15, get_book_odds("Away Team Under 1.5", p_away_under_15), True),
            ("Home Clean Sheet Yes", "Home Clean Sheet: Yes", "Team Goals & Clean Sheets", p_home_cs, get_book_odds("Home Clean Sheet Yes", p_home_cs), False),
            ("Home Clean Sheet No", "Home Clean Sheet: No", "Team Goals & Clean Sheets", p_home_cs_no, get_book_odds("Home Clean Sheet No", p_home_cs_no), True),
            ("Away Clean Sheet Yes", "Away Clean Sheet: Yes", "Team Goals & Clean Sheets", p_away_cs, get_book_odds("Away Clean Sheet Yes", p_away_cs), False),
            ("Away Clean Sheet No", "Away Clean Sheet: No", "Team Goals & Clean Sheets", p_away_cs_no, get_book_odds("Away Clean Sheet No", p_away_cs_no), True),
            ("Home Win Either Half", "Home Team to Win Either Half: Yes", "Team Goals & Clean Sheets", p_home_weh, get_book_odds("Home Win Either Half", p_home_weh), True),
            ("Away Win Either Half", "Away Team to Win Either Half: Yes", "Team Goals & Clean Sheets", p_away_weh, get_book_odds("Away Win Either Half", p_away_weh), True),
            ("Both Halves Under 1.5", "Both Halves Under 1.5: Yes", "Team Goals & Clean Sheets", p_bh_under_15, get_book_odds("Both Halves Under 1.5", p_bh_under_15), True),
        ]

        # If the user's slip selection is a custom/niche SportyBet market not in raw_bookable (e.g. Corners, Bookings, Handicaps),
        # inject it automatically so it can be analyzed, displayed, and re-booked with 100% fidelity!
        user_sel_pick = str(features.get("selected_outcome") or "").strip()
        user_sel_odds = float(features.get("selected_odds") or 0.0)
        existing_picks_lower = {r[0].lower(): r for r in raw_bookable}
        if user_sel_pick and user_sel_pick.lower() not in existing_picks_lower:
            est_prob = max(0.08, min(0.92, (0.93 / user_sel_odds) if user_sel_odds > 1.02 else 0.55))
            raw_bookable.insert(0, (
                user_sel_pick,
                f"Slip Pick: {user_sel_pick}",
                "Extracted Slip Market",
                est_prob,
                user_sel_odds if user_sel_odds > 1.01 else round(0.93 / est_prob, 2),
                est_prob >= 0.65
            ))

        bookable_options = []
        selected_pick_eval = None
        for pick_key, label, category, prob_v, b_odds, is_safe_cat in raw_bookable:
            # If this matches the user's slip pick and user_sel_odds > 1.01, use the exact slip odds
            if user_sel_pick and pick_key.lower() == user_sel_pick.lower() and user_sel_odds > 1.01:
                b_odds = round(user_sel_odds, 2)
            ev = (prob_v * b_odds) - 1.0
            # Composite score balances high win probability, playable odds (>=1.20), and expected value
            playable_bonus = 0.15 if (1.20 <= b_odds <= 2.45) else (-0.25 if b_odds < 1.14 else 0.0)
            score = (prob_v * 0.65) + (max(-0.2, min(0.4, ev)) * 0.35) + playable_bonus
            opt_dict = {
                "pick": pick_key,
                "label": label,
                "category": category,
                "prob": round(prob_v, 3),
                "prob_pct": round(prob_v * 100, 1),
                "odds": b_odds,
                "fair_odds": round(1.0 / prob_v, 2) if prob_v > 0 else 99.0,
                "edge_pct": round(ev * 100, 1),
                "is_safe": is_safe_cat and prob_v >= 0.64,
                "score": round(score, 4)
            }
            bookable_options.append(opt_dict)
            if user_sel_pick and pick_key.lower() == user_sel_pick.lower():
                selected_pick_eval = opt_dict

        # Sort for Top Pick (best balance of probability >= 52%, playable odds >= 1.18, and EV)
        viable_top = [o for o in bookable_options if o["prob"] >= 0.50 and 1.18 <= o["odds"] <= 3.10]
        if not viable_top:
            viable_top = bookable_options
        top_pick = sorted(viable_top, key=lambda x: x["score"], reverse=True)[0]

        # Sort for Safe Pick (Double Chance, Over 1.5, Under 3.5, 1st/2nd Half O/U, DNB with highest probability & odds >= 1.16)
        viable_safe = [o for o in bookable_options if o["is_safe"] and o["odds"] >= 1.16]
        if not viable_safe:
            viable_safe = sorted(bookable_options, key=lambda x: x["prob"], reverse=True)
        else:
            viable_safe = sorted(viable_safe, key=lambda x: (x["prob"], x["edge_pct"]), reverse=True)
        safe_pick = viable_safe[0]

        # Sort for Value Pick (Highest positive EV with prob >= 40%)
        viable_val = [o for o in bookable_options if o["prob"] >= 0.40 and o["odds"] <= 3.60]
        if not viable_val:
            viable_val = bookable_options
        value_pick = sorted(viable_val, key=lambda x: (x["edge_pct"], x["prob"]), reverse=True)[0]

        if not selected_pick_eval:
            selected_pick_eval = top_pick

        return {
            "model_version": "Dixon-Coles Bivariate v4.3 (Full + 1st/2nd Half)",
            "venue_split_xg": {"home": home_xg, "away": away_xg},
            "selected_pick_evaluation": selected_pick_eval,
            "first_half": {
                "Over 0.5": {"prob": round(ht_p["o05"], 3), "fair_odds": round(1.0/ht_p["o05"], 2)},
                "Under 1.5": {"prob": round(ht_p["u15"], 3), "fair_odds": round(1.0/ht_p["u15"], 2)},
                "1X": {"prob": round(ht_p["1x"], 3), "fair_odds": round(1.0/ht_p["1x"], 2)},
                "X2": {"prob": round(ht_p["x2"], 3), "fair_odds": round(1.0/ht_p["x2"], 2)},
            },
            "second_half": {
                "Over 0.5": {"prob": round(sh_p["o05"], 3), "fair_odds": round(1.0/sh_p["o05"], 2)},
                "Over 1.5": {"prob": round(sh_p["o15"], 3), "fair_odds": round(1.0/sh_p["o15"], 2)},
                "Under 1.5": {"prob": round(sh_p["u15"], 3), "fair_odds": round(1.0/sh_p["u15"], 2)},
                "Under 2.5": {"prob": round(sh_p["u25"], 3), "fair_odds": round(1.0/sh_p["u25"], 2)},
            },
            "recommended_markets": {
                "top_pick": top_pick,
                "safe_pick": safe_pick,
                "value_pick": value_pick,
                "selected_pick_eval": selected_pick_eval,
                "bookable_options": bookable_options
            },
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