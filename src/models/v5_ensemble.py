"""
V5 Multi-Model Predictive Ensemble & Uncertainty Quantification Engine.
Stacks Dixon-Coles bivariate projections, de-vigged market baseline probabilities,
and consensus signals while quantifying model divergence and epistemic uncertainty.
"""

import math
from typing import Dict, List, Any, Tuple
import numpy as np
from .v5_dixon_coles import DixonColesEngine
from .v5_market_intelligence import get_market_intelligence
from .v5_calibration import get_calibrator
from .v5_league_intelligence import get_league_intelligence


class ModelEnsembleEngine:
    """
    Ensemble engine that combines statistical physics (Dixon-Coles),
    market-implied equilibrium (SportyBet odds), and supplementary signals.
    Measures inter-model disagreement to quantify uncertainty and prevent false confidence.
    """

    def __init__(self):
        self.dc_engine = DixonColesEngine()
        self.market_intel = get_market_intelligence()
        self.calibrator = get_calibrator()
        self.league_intel = get_league_intelligence()

        # Optimal weights fitted through out-of-sample backtesting
        # Market odds are the strongest informational baseline, complemented by Dixon-Coles
        self.weight_dc = 0.45
        self.weight_market = 0.40
        self.weight_consensus = 0.15

    def predict_match(
        self,
        fixture: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generates calibrated probability distribution, expected goals, market intelligence,
        and uncertainty quantification for a given fixture.
        """
        home = fixture.get("home_team", "Home")
        away = fixture.get("away_team", "Away")
        league = fixture.get("league", "Football")

        oh = float(fixture.get("odds_home", 2.10))
        od = float(fixture.get("odds_draw", 3.20))
        oa = float(fixture.get("odds_away", 3.10))
        m_odds = dict(fixture.get("market_odds") or {})

        # 1. League Intelligence Profile
        league_profile = self.league_intel.classify_league(league)

        # 2. De-vigged Market Probabilities
        market_info = self.market_intel.get_1x2_market_intelligence(oh, od, oa)
        p_mkt_h = market_info["fair_prob_home"]
        p_mkt_d = market_info["fair_prob_draw"]
        p_mkt_a = market_info["fair_prob_away"]

        # 3. Dixon-Coles Bivariate Goal Modeling
        lambda_h, lambda_a = self.dc_engine.estimate_lambdas_from_odds(oh, od, oa, m_odds.get("Over 2.5"))
        dc_markets = self.dc_engine.generate_all_market_probabilities(lambda_h, lambda_a)
        p_dc_h = dc_markets["Home"]
        p_dc_d = dc_markets["Draw"]
        p_dc_a = dc_markets["Away"]

        # 4. Supplementary Consensus (if available)
        p_con_h, p_con_d, p_con_a = p_mkt_h, p_mkt_d, p_mkt_a
        forebet_probs = fixture.get("forebet_probs")
        if forebet_probs:
            import re
            m = re.findall(r'[HDA]:(\d+)%', forebet_probs)
            if len(m) >= 3:
                p_con_h = float(m[0]) / 100.0
                p_con_d = float(m[1]) / 100.0
                p_con_a = float(m[2]) / 100.0

        # 5. Stacking / Ensemble Combination
        raw_h = (self.weight_dc * p_dc_h) + (self.weight_market * p_mkt_h) + (self.weight_consensus * p_con_h)
        raw_d = (self.weight_dc * p_dc_d) + (self.weight_market * p_mkt_d) + (self.weight_consensus * p_con_d)
        raw_a = (self.weight_dc * p_dc_a) + (self.weight_market * p_mkt_a) + (self.weight_consensus * p_con_a)

        # 6. Multiclass Calibration
        cal_h, cal_d, cal_a = self.calibrator.calibrate_1x2_distribution(raw_h, raw_d, raw_a)

        # Apply League Confidence Damping
        damping = league_profile["confidence_damping"]
        if damping < 1.0:
            # Dampen deviation from 0.33 towards neutral distribution
            cal_h = round(0.333 + (cal_h - 0.333) * damping, 4)
            cal_d = round(0.333 + (cal_d - 0.333) * damping, 4)
            cal_a = round(1.0 - (cal_h + cal_d), 4)

        # 7. Model Disagreement / Uncertainty Quantification (Variance across models)
        h_variance = float(np.var([p_dc_h, p_mkt_h, p_con_h]))
        uncertainty_score = round(math.sqrt(h_variance) * 100.0 * league_profile["uncertainty_multiplier"], 2)
        has_high_uncertainty = uncertainty_score > 6.5  # Models diverge by more than 6.5%

        # 8. Calibrate Derivative Markets
        calibrated_markets = {}
        for m_name, raw_p in dc_markets.items():
            if m_name in ("Home", "Draw", "Away", "lambda_home", "lambda_away", "lambda_total"):
                continue
            cal_p = self.calibrator.calibrate_binary_prob(raw_p, m_name)
            # Apply league tier damping
            if damping < 1.0 and cal_p > 0.50:
                cal_p = round(0.50 + (cal_p - 0.50) * damping, 4)
            calibrated_markets[m_name] = cal_p

        calibrated_markets["Home"] = cal_h
        calibrated_markets["Draw"] = cal_d
        calibrated_markets["Away"] = cal_a
        calibrated_markets["lambda_home"] = lambda_h
        calibrated_markets["lambda_away"] = lambda_a
        calibrated_markets["lambda_total"] = dc_markets["lambda_total"]

        return {
            "fixture_name": f"{home} vs {away}",
            "home_team": home,
            "away_team": away,
            "league": league,
            "league_tier": league_profile["tier"],
            "league_data_quality": league_profile["data_quality_score"],
            "probabilities": calibrated_markets,
            "market_intelligence": market_info,
            "uncertainty_score": uncertainty_score,
            "has_high_uncertainty": has_high_uncertainty,
            "model_agreement_rating": "Strong" if uncertainty_score <= 3.5 else ("Moderate" if uncertainty_score <= 6.5 else "Divergent"),
            "auto_bet_allowed": league_profile["auto_bet_allowed"] and not has_high_uncertainty
        }


# Singleton accessor
_ensemble_instance = None

def get_ensemble_engine() -> ModelEnsembleEngine:
    global _ensemble_instance
    if _ensemble_instance is None:
        _ensemble_instance = ModelEnsembleEngine()
    return _ensemble_instance
