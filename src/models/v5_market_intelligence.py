"""
V5 Market Probability Intelligence & Expected Value (EV) Engine.
Implements bookmaker margin removal (proportional & power de-vigging),
computes fair market prices, calculates true Expected Value (EV),
tracks Closing Line Value (CLV), and derives Quarter-Kelly bankroll stake fractions.
"""

import math
from typing import Dict, List, Tuple, Any, Optional


class MarketIntelligenceEngine:
    """
    Analyzes bookmaker betting markets, extracts fair de-vigged market probabilities,
    quantifies predictive model divergence, and calculates mathematical Expected Value.
    """

    def __init__(self):
        pass

    def remove_margin_proportional(self, odds_list: List[float]) -> List[float]:
        """
        Removes bookmaker overround (margin) proportionally across an odds list.
        P_fair_i = (1 / O_i) / sum(1 / O_j)
        """
        valid_odds = [max(1.01, float(o)) for o in odds_list]
        inv_probs = [1.0 / o for o in valid_odds]
        overround = sum(inv_probs)
        if overround <= 0:
            return [1.0 / len(odds_list)] * len(odds_list)
        return [round(p / overround, 4) for p in inv_probs]

    def remove_margin_power(self, odds_list: List[float], beta: float = 1.15) -> List[float]:
        """
        Removes margin with power transformation to correct for favorite-longshot bias.
        Longshots generally carry higher proportional margins than heavy favorites.
        """
        valid_odds = [max(1.01, float(o)) for o in odds_list]
        pow_probs = [(1.0 / o) ** beta for o in valid_odds]
        tot = sum(pow_probs)
        if tot <= 0:
            return self.remove_margin_proportional(odds_list)
        return [round(p / tot, 4) for p in pow_probs]

    def get_1x2_market_intelligence(
        self,
        odds_h: float,
        odds_d: float,
        odds_a: float
    ) -> Dict[str, Any]:
        """
        Extracts fair probabilities, bookmaker overround, and margin-free fair odds for a 1X2 market.
        """
        raw_odds = [odds_h, odds_d, odds_a]
        inv_sum = sum(1.0 / max(1.01, o) for o in raw_odds)
        overround_pct = round((inv_sum - 1.0) * 100, 2)
        fair_probs = self.remove_margin_proportional(raw_odds)

        return {
            "overround_pct": overround_pct,
            "bookmaker_margin_pct": round(overround_pct / inv_sum, 2),
            "fair_prob_home": fair_probs[0],
            "fair_prob_draw": fair_probs[1],
            "fair_prob_away": fair_probs[2],
            "fair_odds_home": round(1.0 / fair_probs[0], 2) if fair_probs[0] > 0 else 99.0,
            "fair_odds_draw": round(1.0 / fair_probs[1], 2) if fair_probs[1] > 0 else 99.0,
            "fair_odds_away": round(1.0 / fair_probs[2], 2) if fair_probs[2] > 0 else 99.0,
        }

    def compute_expected_value(
        self,
        calibrated_prob: float,
        market_odds: float
    ) -> Dict[str, Any]:
        """
        Computes mathematically rigorous Expected Value:
        EV = (P_calibrated * Odds) - 1.0
        EV_pct = EV * 100%
        """
        p = max(0.0, min(1.0, float(calibrated_prob)))
        o = max(1.0, float(market_odds))
        ev = (p * o) - 1.0
        ev_pct = round(ev * 100, 2)
        is_positive_ev = ev > 0.015  # At least 1.5% edge over the market

        return {
            "expected_value": round(ev, 4),
            "ev_pct": ev_pct,
            "is_positive_ev": is_positive_ev,
            "break_even_prob": round((1.0 / o) * 100, 2),
            "model_prob": round(p * 100, 2),
            "edge_pct": round((p - (1.0 / o)) * 100, 2)
        }

    def compute_clv(
        self,
        taken_odds: float,
        closing_odds: float
    ) -> Dict[str, Any]:
        """
        Calculates Closing Line Value (CLV):
        CLV_odds = (O_taken / O_closing) - 1.0
        Positive CLV is the primary empirical proof of beating the bookmaker market.
        """
        t = max(1.01, float(taken_odds))
        c = max(1.01, float(closing_odds))
        clv = (t / c) - 1.0
        clv_pct = round(clv * 100, 2)
        beat_closing = clv > 0.0

        return {
            "taken_odds": t,
            "closing_odds": c,
            "clv": round(clv, 4),
            "clv_pct": clv_pct,
            "beat_closing": beat_closing
        }

    def calculate_quarter_kelly_stake(
        self,
        calibrated_prob: float,
        market_odds: float,
        bankroll: float,
        max_stake_cap_pct: float = 0.05
    ) -> Dict[str, Any]:
        """
        Computes Quarter-Kelly bankroll stake fraction:
        b = Odds - 1.0
        f* = (b*p - (1 - p)) / b
        f_quarter = 0.25 * f* (if f* > 0 else 0.0)
        Enforces maximum safety stake ceiling (default 5% of bankroll).
        """
        p = max(0.0, min(1.0, float(calibrated_prob)))
        o = max(1.01, float(market_odds))
        b = o - 1.0

        f_star = (b * p - (1.0 - p)) / b if b > 0 else 0.0
        if f_star <= 0.0:
            return {
                "f_star": 0.0,
                "f_quarter": 0.0,
                "recommended_stake_ngn": 0.0,
                "stake_pct_of_bankroll": 0.0,
                "qualifies_for_wager": False
            }

        f_quarter = min(max_stake_cap_pct, 0.25 * f_star)
        recommended_stake = round(bankroll * f_quarter, 2)

        return {
            "f_star": round(f_star, 4),
            "f_quarter": round(f_quarter, 4),
            "recommended_stake_ngn": recommended_stake,
            "stake_pct_of_bankroll": round(f_quarter * 100, 2),
            "qualifies_for_wager": recommended_stake >= 10.0
        }


# Singleton accessor
_market_intel_instance = None

def get_market_intelligence() -> MarketIntelligenceEngine:
    global _market_intel_instance
    if _market_intel_instance is None:
        _market_intel_instance = MarketIntelligenceEngine()
    return _market_intel_instance
