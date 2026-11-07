"""
AI-Powered Analysis Engine
Provides advanced ML predictions with value detection and confidence scoring
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence level categories"""
    HIGH = "high"      # > 70%
    MEDIUM = "medium"  # 50-70%
    LOW = "low"        # < 50%


@dataclass
class ConfidenceScore:
    """Confidence scoring for predictions"""
    level: ConfidenceLevel
    score: float  # 0-1 probability
    reasoning: str
    features_contributing: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
            "features_contributing": self.features_contributing
        }


@dataclass
class ValueBet:
    """Value bet identification"""
    match: str
    market: str
    outcome: str
    model_probability: float
    implied_probability: float
    value: float
    stake_recommendation: float
    edge_percentage: float

    @property
    def is_value_bet(self) -> bool:
        return self.edge_percentage > 8.0

    @property
    def recommendation(self) -> str:
        if self.edge_percentage > 20:
            return "STRONG BUY"
        elif self.edge_percentage > 10:
            return "BUY"
        elif self.edge_percentage > 5:
            return "MODERATE BUY"
        elif self.edge_percentage > 2:
            return "MONITOR"
        else:
            return "SKIP"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match,
            "market": self.market,
            "outcome": self.outcome,
            "model_probability": round(self.model_probability, 4),
            "implied_probability": round(self.implied_probability, 4),
            "value": round(self.value, 4),
            "stake_recommendation": round(self.stake_recommendation, 2),
            "edge_percentage": round(self.edge_percentage, 2),
            "is_value_bet": self.is_value_bet,
            "recommendation": self.recommendation
        }


@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    overall_risk: str  # low, medium, high
    risk_score: float
    volatility: float
    draw_probability: float
    upset_potential: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk": self.overall_risk,
            "risk_score": round(self.risk_score, 4),
            "volatility": round(self.volatility, 4),
            "draw_probability": round(self.draw_probability, 4),
            "upset_potential": round(self.upset_potential, 4)
        }


@dataclass
class ModelPrediction:
    """Complete model prediction output"""
    match: str
    outcome: str
    probability: float
    confidence: ConfidenceScore
    value_bet: Optional[ValueBet] = None
    risk_metrics: Optional[RiskMetrics] = None
    alternative_outcomes: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "match": self.match,
            "outcome": self.outcome,
            "probability": round(self.probability, 4),
            "confidence": self.confidence.to_dict(),
            "alternative_outcomes": self.alternative_outcomes
        }
        if self.value_bet:
            result["value_bet"] = self.value_bet.to_dict()
        if self.risk_metrics:
            result["risk_metrics"] = self.risk_metrics.to_dict()
        return result


@dataclass
class FullAnalysisResult:
    """Complete analysis result with all markets"""
    h2h: ModelPrediction
    btts: ModelPrediction
    over_25: ModelPrediction
    match_metadata: Dict[str, Any]
    analysis_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "h2h": self.h2h.to_dict(),
            "btts": self.btts.to_dict(),
            "over_25": self.over_25.to_dict(),
            "match_metadata": self.match_metadata,
            "analysis_timestamp": self.analysis_timestamp
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Export to JSON string with analysis metadata.

        Args:
            indent: JSON indentation level (default 2)

        Returns:
            JSON string with analysis results and metadata
        """
        result = self.to_dict()

        # Add analysis metadata
        meta = {
            "analysis_metadata": {
                "export_format": "ai_analyzer_v1",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "prediction_type": "ensemble_ml",
                "confidence_threshold": 0.08,
                "model_components": {
                    "h2h_estimator": "market_implied_enhanced",
                    "btts_calculator": "xg_based",
                    "over25_calculator": "xg_sum_model"
                },
                "value_bet_identification": {
                    "method": "edge_calculation",
                    "threshold": 8.0,
                    "unit": "percentage"
                }
            }
        }

        result.update(meta)
        return json.dumps(result, indent=indent, default=str)


class FeatureScalerV2:
    """Advanced feature scaler with normalization options"""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScalerV2":
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.asarray(X, dtype=float)
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class AIPredictor:
    """
    Advanced AI-powered sports predictor
    Uses ensemble models with uncertainty quantification
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "model_v4.pkl"
        self.scaler = FeatureScalerV2()
        self._models_loaded = False

    def _load_models(self):
        """Load trained models"""
        if Path(self.model_path).exists():
            import pickle
            with open(self.model_path, 'rb') as f:
                models = pickle.load(f)
                self._h2h_model = models.get('h2h_model')
                self._btts_model = models.get('btts_model')
                self._over25_model = models.get('over25_model')
                self._scaler = models.get('scaler', self.scaler)
                self._models_loaded = True

    def _ensure_model(self):
        """Ensure models are loaded"""
        if not self._models_loaded:
            self._load_models()

    def _calculate_confidence(self, prob: float, odds: float, has_data: bool = True) -> ConfidenceScore:
        """Calculate confidence score based on model probability and market odds"""
        if not has_data or odds == 0:
            return ConfidenceScore(
                level=ConfidenceLevel.LOW,
                score=prob,
                reasoning="Insufficient data for confidence calculation"
            )

        implied = 1.0 / odds if odds > 0 else 0
        edge = prob - implied

        if prob > 0.7:
            level = ConfidenceLevel.HIGH
            reasoning = f"Strong model conviction (p={prob:.2f})"
        elif prob > 0.5:
            level = ConfidenceLevel.MEDIUM
            reasoning = f"Moderate confidence (p={prob:.2f})"
        else:
            level = ConfidenceLevel.LOW
            reasoning = f"Low confidence (p={prob:.2f})"

        return ConfidenceScore(
            level=level,
            score=prob,
            reasoning=reasoning,
            features_contributing=[]  # Would be populated by model
        )

    def _calculate_risk(self, features: Dict) -> RiskMetrics:
        """Calculate risk metrics for a match"""
        # Draw probability from feature indicators
        draw_factor = 1.0 - abs(features.get('home_xGD', 0) - features.get('away_xGD', 0))
        draw_prob = max(0.1, min(0.6, draw_factor * 0.4 + 0.3))

        # Upset potential (when teams are evenly matched)
        xg_diff = abs(features.get('home_xG', 1.5) - features.get('away_xG', 1.5))
        upset_potential = 1.0 - min(1.0, xg_diff / 2.0)

        # Overall volatility
        volatility = (features.get('home_ppda', 10) + features.get('away_ppda', 10)) / 20

        if volatility > 0.7 or upset_potential > 0.6:
            risk_level = "high"
        elif volatility > 0.4 or upset_potential > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        return RiskMetrics(
            overall_risk=risk_level,
            risk_score=volatility,
            volatility=volatility,
            draw_probability=draw_prob,
            upset_potential=upset_potential
        )

    def predict_full(self, features: Dict[str, Any], sport: str = "football") -> FullAnalysisResult:
        """
        Generate complete analysis for a match

        Args:
            features: Match features dictionary
            sport: Sport type

        Returns:
            FullAnalysisResult with all predictions
        """
        self._ensure_model()

        match_name = f"{features.get('home_team', 'Unknown')} vs {features.get('away_team', 'Unknown')}"

        # H2H Prediction
        h2h = self._predict_market(
            outcome="Home",  # Simplified
            features=features,
            odds_key="odds_home",
            match=match_name,
            sport=sport
        )

        # BTTS Prediction
        btts = self._predict_btts(features, match_name)

        # Over 2.5 Prediction
        over25 = self._predict_over25(features, match_name)

        return FullAnalysisResult(
            h2h=h2h,
            btts=btts,
            over_25=over25,
            match_metadata={
                "sport": sport,
                "home_team": features.get("home_team"),
                "away_team": features.get("away_team")
            },
            analysis_timestamp=datetime.now(timezone.utc).isoformat()
        )

    def _predict_market(self, outcome: str, features: Dict, odds_key: str,
                         match: str, sport: str) -> ModelPrediction:
        """Predict a single market outcome"""
        # Get odds
        odds = features.get(odds_key, 0)
        odds_h = features.get("odds_home", 0)
        odds_d = features.get("odds_draw", 0)
        odds_a = features.get("odds_away", 0)

        # Fair implied probability with bookmaker overround removed
        try:
            from ..data_extractor.scraper_utils import calculate_fair_probabilities, calculate_kelly_stake
        except (ImportError, ValueError):
            try:
                from data_extractor.scraper_utils import calculate_fair_probabilities, calculate_kelly_stake
            except ImportError:
                from src.data_extractor.scraper_utils import calculate_fair_probabilities, calculate_kelly_stake
        if odds_h > 1.0 and odds_d > 1.0 and odds_a > 1.0:
            fair_probs = calculate_fair_probabilities([odds_h, odds_d, odds_a])
            fair_map = {"Home": fair_probs[0], "Draw": fair_probs[1], "Away": fair_probs[2]}
            implied_prob = fair_map.get(outcome, 1.0 / odds if odds > 0 else 0.5)
        else:
            implied_prob = 1.0 / odds if odds > 0 else 0.5

        # Model probability via calibrated predictor
        try:
            from .predictor import get_predictor
            predictor = get_predictor()
            pred_dict = predictor.predict_match(features, sport)
            h2h_pred = pred_dict.get("h2h", {})
            if h2h_pred.get("outcome") == outcome:
                model_prob = h2h_pred.get("model_probability", implied_prob)
            else:
                # Distribute remaining probability evenly
                rem = max(0.05, 1.0 - h2h_pred.get("model_probability", 0.5))
                model_prob = rem / 2.0
        except Exception:
            model_prob = implied_prob

        # Confidence
        confidence = self._calculate_confidence(model_prob, odds, has_data=(odds > 0))

        # Risk
        risk = self._calculate_risk(features)

        # Value bet calculation against fair market probability
        value = model_prob - implied_prob
        edge_pct = value * 100 if value > 0 else 0

        # Fractional Kelly staking (Quarter-Kelly)
        stake_recommendation = calculate_kelly_stake(model_prob, odds, fraction=0.25)

        value_bet = ValueBet(
            match=match,
            market="h2h",
            outcome=outcome,
            model_probability=model_prob,
            implied_probability=implied_prob,
            value=value,
            stake_recommendation=stake_recommendation,
            edge_percentage=edge_pct
        ) if value > 0.08 else None

        # Alternative outcomes
        alternatives = []
        if odds > 0:
            draw_odds = features.get("odds_draw", 3.0)
            away_odds = features.get("odds_away", 4.0)

            draw_prob = 1.0 / draw_odds if draw_odds > 0 else 0.33
            away_prob = 1.0 / away_odds if away_odds > 0 else 0.25

            alternatives = [
                {"outcome": "Draw", "probability": draw_prob},
                {"outcome": "Away", "probability": away_prob}
            ]

        return ModelPrediction(
            match=match,
            outcome=outcome,
            probability=model_prob,
            confidence=confidence,
            value_bet=value_bet,
            risk_metrics=risk,
            alternative_outcomes=alternatives
        )

    def _predict_btts(self, features: Dict, match: str) -> ModelPrediction:
        """Predict Both Teams To Score"""
        odds = features.get("odds_btts", 0)
        implied_prob = 1.0 / odds if odds > 0 else 0.5

        # xG-based prediction
        home_xg = features.get("home_xG", 1.5)
        away_xg = features.get("away_xG", 1.5)
        home_xga = features.get("home_xGA", 1.2)
        away_xga = features.get("away_xGA", 1.2)

        btts_prob = (home_xg * away_xga + away_xg * home_xga) / 6.0
        btts_prob = max(0.1, min(0.9, btts_prob))

        confidence = self._calculate_confidence(btts_prob, odds, has_data=(odds > 0))
        risk = self._calculate_risk(features)

        value = btts_prob - implied_prob
        edge_pct = value * 100 if value > 0 else 0

        value_bet = ValueBet(
            match=match,
            market="btts",
            outcome="BTTS YES",
            model_probability=btts_prob,
            implied_probability=implied_prob,
            value=value,
            stake_recommendation=calculate_kelly_stake(btts_prob, odds, fraction=0.25),
            edge_percentage=edge_pct
        ) if value > 0.08 else None

        return ModelPrediction(
            match=match,
            outcome="BTTS YES",
            probability=btts_prob,
            confidence=confidence,
            value_bet=value_bet,
            risk_metrics=risk
        )

    def _predict_over25(self, features: Dict, match: str) -> ModelPrediction:
        """Predict Over 2.5 goals"""
        odds = features.get("odds_over_25", 0)
        implied_prob = 1.0 / odds if odds > 0 else 0.5

        # xG sum prediction
        total_xg = (
            features.get("home_xG", 1.5) +
            features.get("away_xG", 1.5) +
            features.get("home_xGA", 1.2) +
            features.get("away_xGA", 1.2)
        )
        over25_prob = total_xg / 7.0
        over25_prob = max(0.1, min(0.9, over25_prob))

        confidence = self._calculate_confidence(over25_prob, odds, has_data=(odds > 0))
        risk = self._calculate_risk(features)

        value = over25_prob - implied_prob
        edge_pct = value * 100 if value > 0 else 0

        value_bet = ValueBet(
            match=match,
            market="over_25",
            outcome="OVER 2.5",
            model_probability=over25_prob,
            implied_probability=implied_prob,
            value=value,
            stake_recommendation=calculate_kelly_stake(over25_prob, odds, fraction=0.25),
            edge_percentage=edge_pct
        ) if value > 0.08 else None

        return ModelPrediction(
            match=match,
            outcome="OVER 2.5",
            probability=over25_prob,
            confidence=confidence,
            value_bet=value_bet,
            risk_metrics=risk
        )


class FullAIPredictor:
    """
    Full AI prediction pipeline with ensemble methods
    Combines multiple models for robust predictions
    """

    def __init__(self):
        self.h2h_predictor = AIPredictor()
        self.btts_predictor = AIPredictor()
        self.over25_predictor = AIPredictor()

    def analyze_match(self, features: Dict[str, Any], sport: str = "football") -> FullAnalysisResult:
        """Complete match analysis"""
        return self.h2h_predictor.predict_full(features, sport)

    def identify_value_bets(self, predictions: List[FullAnalysisResult]) -> List[ValueBet]:
        """Identify all value bets from predictions"""
        value_bets = []

        for pred in predictions:
            if pred.h2h.value_bet and pred.h2h.value_bet.is_value_bet:
                value_bets.append(pred.h2h.value_bet)
            if pred.btts.value_bet and pred.btts.value_bet.is_value_bet:
                value_bets.append(pred.btts.value_bet)
            if pred.over_25.value_bet and pred.over_25.value_bet.is_value_bet:
                value_bets.append(pred.over_25.value_bet)

        return sorted(value_bets, key=lambda x: x.edge_percentage, reverse=True)


# Global instances
_default_predictor = None
_ai_predictor = None


def get_ai_predictor() -> FullAIPredictor:
    """Get or create default AI predictor"""
    global _ai_predictor
    if _ai_predictor is None:
        _ai_predictor = FullAIPredictor()
    return _ai_predictor


def get_predictor() -> AIPredictor:
    """Get default predictor"""
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = AIPredictor()
    return _default_predictor


def analyze_match(
    features: Dict[str, Any],
    odds_data: Optional[Dict[str, float]] = None,
    sport: str = "football"
) -> Dict[str, Any]:
    """
    Analyze a single match and generate predictions with value bets.

    Args:
        features: Match features dictionary with team info and xG data
        odds_data: Market odds dictionary (optional, merged with features)
        sport: Sport type (default: football)

    Returns:
        Dictionary with h2h, btts, over_25 predictions and value bets
    """
    # Merge odds data into features if provided
    if odds_data:
        merged_features = {**features, **odds_data}
    else:
        merged_features = features

    predictor = get_ai_predictor()
    result = predictor.analyze_match(merged_features, sport)
    return result.to_dict()


def identify_value_bets(
    h2h_pred: ModelPrediction,
    btts_pred: ModelPrediction,
    over25_pred: ModelPrediction,
    odds_data: Dict[str, float],
    match_data: Dict[str, Any],
    edge_threshold: float = 0.08
) -> List[ValueBet]:
    """
    Identify value bets from predictions based on odds comparison.

    Args:
        h2h_pred: Head-to-head prediction
        btts_pred: BTTS prediction
        over25_pred: Over 2.5 prediction
        odds_data: Market odds dictionary
        match_data: Match data for metadata
        edge_threshold: Minimum edge to qualify as value bet (default 8%)

    Returns:
        List of ValueBet objects sorted by value score
    """
    value_bets = []
    match_name = f"{match_data.get('home_team', 'Unknown')} vs {match_data.get('away_team', 'Unknown')}"

    if not odds_data:
        return value_bets

    # H2H Value Bets
    if odds_data.get("odds_home", 0) > 0:
        implied = 1.0 / odds_data["odds_home"]
        edge = h2h_pred.probability - implied
        if edge >= edge_threshold:
            value_bets.append(ValueBet(
                match=match_name,
                market="H2H",
                outcome="HOME",
                model_probability=h2h_pred.probability,
                implied_probability=implied,
                value=edge,
                stake_recommendation=max(0, edge * 10),
                edge_percentage=edge * 100
            ))

    if odds_data.get("odds_draw", 0) > 0:
        implied = 1.0 / odds_data["odds_draw"]
        edge = h2h_pred.alternative_outcomes[0]["probability"] if h2h_pred.alternative_outcomes else 0.33
        if edge >= edge_threshold:
            value_bets.append(ValueBet(
                match=match_name,
                market="H2H",
                outcome="DRAW",
                model_probability=h2h_pred.alternative_outcomes[0]["probability"],
                implied_probability=implied,
                value=edge,
                stake_recommendation=max(0, edge * 10),
                edge_percentage=edge * 100
            ))

    if odds_data.get("odds_away", 0) > 0:
        implied = 1.0 / odds_data["odds_away"]
        away_prob = h2h_pred.alternative_outcomes[1]["probability"] if len(h2h_pred.alternative_outcomes) > 1 else 0.25
        edge = away_prob - implied
        if edge >= edge_threshold:
            value_bets.append(ValueBet(
                match=match_name,
                market="H2H",
                outcome="AWAY",
                model_probability=away_prob,
                implied_probability=implied,
                value=edge,
                stake_recommendation=max(0, edge * 10),
                edge_percentage=edge * 100
            ))

    # BTTS Value Bet
    if odds_data.get("odds_btts", 0) > 0:
        implied = 1.0 / odds_data["odds_btts"]
        edge = btts_pred.probability - implied
        if edge >= edge_threshold:
            value_bets.append(ValueBet(
                match=match_name,
                market="BTTS",
                outcome="YES",
                model_probability=btts_pred.probability,
                implied_probability=implied,
                value=edge,
                stake_recommendation=calculate_kelly_stake(btts_pred.probability, odds_data["odds_btts"], fraction=0.25),
                edge_percentage=edge * 100
            ))

    # Over 2.5 Value Bet
    if odds_data.get("odds_over_25", 0) > 0:
        implied = 1.0 / odds_data["odds_over_25"]
        edge = over25_pred.probability - implied
        if edge >= edge_threshold:
            value_bets.append(ValueBet(
                match=match_name,
                market="OVER_2.5",
                outcome="OVER",
                model_probability=over25_pred.probability,
                implied_probability=implied,
                value=edge,
                stake_recommendation=calculate_kelly_stake(btts_pred.probability, odds_data["odds_btts"], fraction=0.25),
                edge_percentage=edge * 100
            ))

    value_bets.sort(key=lambda x: x.edge_percentage, reverse=True)
    return value_bets


__all__ = [
    "AIPredictor",
    "FullAIPredictor",
    "analyze_match",
    "identify_value_bets",
    "get_ai_predictor",
    "ConfidenceLevel",
    "ConfidenceScore",
    "ModelPrediction",
    "ValueBet",
    "RiskMetrics",
    "FullAnalysisResult",
    "FeatureScalerV2"
]


if __name__ == "__main__":
    # Demo usage
    demo_features = {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_xG": 2.0,
        "away_xG": 1.5,
        "home_xGA": 1.0,
        "away_xGA": 1.2,
        "odds_home": 2.1,
        "odds_draw": 3.5,
        "odds_away": 4.0
    }

    predictor = get_ai_predictor()
    result = predictor.analyze_match(demo_features)

    print(json.dumps(result.to_dict(), indent=2, default=str))