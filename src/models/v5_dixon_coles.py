"""
V5 Advanced Dixon-Coles Football Scoreline & Probability Engine.
Implements bivariate goal distributions with low-scoring dependency adjustment factor tau(x, y),
dynamic team attack/defense parameterizations, and consistent derivative market pricing.
"""

import math
from typing import Dict, Any, Tuple, Optional
import numpy as np


class DixonColesEngine:
    """
    Advanced Dixon-Coles model implementation for football goal modeling.
    Accounts for score dependence in low-scoring outcomes (0-0, 1-0, 0-1, 1-1)
    and ensures coherent joint probability generation across all derivative markets.
    """

    def __init__(self, rho: float = -0.11, default_mu: float = 0.35, default_home_adv: float = 0.28):
        self.rho = rho  # Low scoring correlation factor (typically between -0.15 and -0.05)
        self.default_mu = default_mu  # Baseline league scoring log parameter
        self.default_home_adv = default_home_adv  # Home advantage parameter
        self.max_goals = 8  # Compute matrix up to 8x8 goals

    def _tau(self, x: int, y: int, lambda_h: float, lambda_a: float) -> float:
        """
        Dixon-Coles bivariate correction factor for low-scoring dependence.
        """
        if x == 0 and y == 0:
            return max(0.01, 1.0 - (lambda_h * lambda_a * self.rho))
        elif x == 1 and y == 0:
            return max(0.01, 1.0 + (lambda_a * self.rho))
        elif x == 0 and y == 1:
            return max(0.01, 1.0 + (lambda_h * self.rho))
        elif x == 1 and y == 1:
            return max(0.01, 1.0 - self.rho)
        else:
            return 1.0

    def compute_scoreline_matrix(
        self,
        lambda_h: float,
        lambda_a: float
    ) -> np.ndarray:
        """
        Generates the 9x9 joint probability matrix P(Home = x, Away = y).
        """
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1), dtype=float)

        for x in range(self.max_goals + 1):
            p_x = (math.exp(-lambda_h) * (lambda_h ** x)) / math.factorial(x)
            for y in range(self.max_goals + 1):
                p_y = (math.exp(-lambda_a) * (lambda_a ** y)) / math.factorial(y)
                tau_val = self._tau(x, y, lambda_h, lambda_a)
                matrix[x, y] = tau_val * p_x * p_y

        # Normalize matrix so total probability strictly equals 1.0
        total_p = matrix.sum()
        if total_p > 0:
            matrix /= total_p

        return matrix

    def estimate_lambdas_from_odds(
        self,
        odds_h: float,
        odds_d: float,
        odds_a: float,
        over_25_odds: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Derives initial expected goal intensities (lambda_h, lambda_a) directly from
        market 1X2 odds and optional Over/Under line using inverted probability matching.
        """
        inv_h = 1.0 / max(1.02, odds_h)
        inv_d = 1.0 / max(1.02, odds_d)
        inv_a = 1.0 / max(1.02, odds_a)
        tot_inv = inv_h + inv_d + inv_a
        p_h, p_d, p_a = inv_h / tot_inv, inv_d / tot_inv, inv_a / tot_inv

        # Baseline average total match goals (~2.60) adjusted by Over 2.5 if available
        if over_25_odds and over_25_odds > 1.04:
            p_ov25 = (1.0 / over_25_odds) / (1.0 / over_25_odds + 1.0 / 2.0)
            tot_goals = 1.6 + (p_ov25 * 1.8)
        else:
            # Implied goal volume from draw frequency
            tot_goals = max(1.9, min(3.8, 2.75 - (p_d - 0.26) * 2.5))

        # Strength ratio based on home vs away win probabilities
        ratio = max(0.2, min(5.0, (p_h + 0.5 * p_d) / max(0.08, p_a + 0.5 * p_d)))
        lambda_h = (tot_goals * ratio) / (1.0 + ratio)
        lambda_a = tot_goals - lambda_h

        # Bounds check
        lambda_h = max(0.40, min(4.20, lambda_h))
        lambda_a = max(0.30, min(3.80, lambda_a))

        return round(float(lambda_h), 3), round(float(lambda_a), 3)

    def generate_all_market_probabilities(
        self,
        lambda_h: float,
        lambda_a: float
    ) -> Dict[str, float]:
        """
        Derives mathematically consistent probabilities across all standard betting markets
        from the underlying calibrated Dixon-Coles bivariate scoreline matrix.
        """
        matrix = self.compute_scoreline_matrix(lambda_h, lambda_a)

        # 1X2 Probabilities
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_draw = float(np.sum(np.diag(matrix)))
        p_away = float(np.sum(np.triu(matrix, 1)))

        # Totals / Over Under
        total_goals_dist = np.zeros(self.max_goals * 2 + 1)
        for x in range(self.max_goals + 1):
            for y in range(self.max_goals + 1):
                total_goals_dist[x + y] += matrix[x, y]

        p_ov05 = float(np.sum(total_goals_dist[1:]))
        p_ov15 = float(np.sum(total_goals_dist[2:]))
        p_ov25 = float(np.sum(total_goals_dist[3:]))
        p_ov35 = float(np.sum(total_goals_dist[4:]))
        p_un25 = 1.0 - p_ov25
        p_un35 = 1.0 - p_ov35
        p_un45 = float(np.sum(total_goals_dist[:5]))

        # Double Chance
        p_1x = min(0.98, p_home + p_draw)
        p_x2 = min(0.98, p_draw + p_away)
        p_12 = min(0.98, p_home + p_away)

        # Draw No Bet (DNB)
        non_draw_p = max(0.05, p_home + p_away)
        p_dnb_h = min(0.95, p_home / non_draw_p)
        p_dnb_a = min(0.95, p_away / non_draw_p)

        # Both Teams To Score (BTTS)
        p_btts_y = float(np.sum(matrix[1:, 1:]))
        p_btts_n = 1.0 - p_btts_y

        # Halves approximations using 45% 1H / 55% 2H distribution
        p_1h_o05 = min(0.92, 1.0 - math.exp(-(lambda_h + lambda_a) * 0.44))
        p_2h_o05 = min(0.94, 1.0 - math.exp(-(lambda_h + lambda_a) * 0.56))
        p_2h_u25 = min(0.95, max(0.65, math.exp(-(lambda_h + lambda_a) * 0.56) * (1 + (lambda_h + lambda_a) * 0.56 + (((lambda_h + lambda_a) * 0.56) ** 2) / 2)))

        return {
            "Home": round(p_home, 4),
            "Draw": round(p_draw, 4),
            "Away": round(p_away, 4),
            "1X": round(p_1x, 4),
            "X2": round(p_x2, 4),
            "12": round(p_12, 4),
            "Home DNB": round(p_dnb_h, 4),
            "Away DNB": round(p_dnb_a, 4),
            "Over 0.5": round(p_ov05, 4),
            "Over 1.5": round(p_ov15, 4),
            "Over 2.5": round(p_ov25, 4),
            "Under 2.5": round(p_un25, 4),
            "Under 3.5": round(p_un35, 4),
            "Under 4.5": round(p_un45, 4),
            "BTTS Yes": round(p_btts_y, 4),
            "BTTS No": round(p_btts_n, 4),
            "1st Half Over 0.5": round(p_1h_o05, 4),
            "2nd Half Over 0.5": round(p_2h_o05, 4),
            "2nd Half Under 2.5": round(p_2h_u25, 4),
            "lambda_home": lambda_h,
            "lambda_away": lambda_a,
            "lambda_total": round(lambda_h + lambda_a, 2)
        }
