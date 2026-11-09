"""
SportyBet One-Cut & System Bet Optimizer
Calculates insurance guarantees for accumulators where 1 leg fails (One-Cut),
and generates multi-combination system bets (e.g. 4/5, 3/5, Yankee, Canadian)
to protect bankrolls against heartbreaking single-game accumulator upsets.
"""

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple


class SystemBetOptimizer:
    """
    Optimizes system combinations and One-Cut hedging for sports accumulators.
    """

    def calculate_one_cut_coverage(
        self,
        legs: List[Dict[str, Any]],
        stake: float = 100.0,
        one_cut_discount_pct: float = 0.10
    ) -> Dict[str, Any]:
        """
        Calculates the exact outcomes under SportyBet One-Cut insurance.
        """
        if len(legs) < 3:
            return {
                "eligible": False,
                "message": "SportyBet One-Cut requires at least 3 distinct accumulator legs."
            }

        odds_list = [float(l.get("selected_odds") or l.get("odds") or 1.5) for l in legs]
        total_odds = 1.0
        for o in odds_list:
            total_odds *= o

        full_payout = round(stake * total_odds, 2)

        # Scenarios where exactly ONE leg loses
        one_loss_payouts = []
        for i, o_lost in enumerate(odds_list):
            rem_odds = 1.0
            for j, o in enumerate(odds_list):
                if i != j:
                    rem_odds *= o

            # One-cut formula: remaining odds * stake * (1.0 - discount)
            one_cut_payout = round(rem_odds * stake * (1.0 - one_cut_discount_pct), 2)
            one_loss_payouts.append({
                "lost_leg_index": i,
                "lost_match": f"{legs[i].get('home_team', '')} vs {legs[i].get('away_team', '')}",
                "lost_odds": o_lost,
                "remaining_odds": round(rem_odds, 2),
                "payout": one_cut_payout,
                "profit": round(one_cut_payout - stake, 2)
            })

        min_payout = min(x["payout"] for x in one_loss_payouts)
        max_one_loss_payout = max(x["payout"] for x in one_loss_payouts)

        return {
            "eligible": True,
            "total_legs": len(legs),
            "stake": stake,
            "full_win_odds": round(total_odds, 2),
            "full_win_payout": full_payout,
            "one_cut_discount_pct": round(one_cut_discount_pct * 100, 1),
            "worst_case_one_cut_payout": min_payout,
            "best_case_one_cut_payout": max_one_loss_payout,
            "worst_case_roi_pct": round(((min_payout - stake) / stake * 100), 1),
            "scenarios": one_loss_payouts
        }

    def generate_system_combinations(
        self,
        legs: List[Dict[str, Any]],
        system_type: str = "4_of_5",
        total_stake: float = 100.0
    ) -> Dict[str, Any]:
        """
        Builds full system coverage combinations (e.g. 4/5, 3/5, 2/3).
        """
        n = len(legs)
        if "_" in system_type:
            parts = system_type.split("_")
            k = int(parts[0])
        else:
            k = max(2, n - 1)

        if k >= n:
            k = max(1, n - 1)

        combos = list(itertools.combinations(range(n), k))
        total_combos = len(combos)
        unit_stake = round(total_stake / total_combos, 2)

        combo_details = []
        payouts = []
        for combo_indices in combos:
            combo_odds = 1.0
            for idx in combo_indices:
                odd = float(legs[idx].get("selected_odds") or legs[idx].get("odds") or 1.5)
                combo_odds *= odd
            pot_return = round(unit_stake * combo_odds, 2)
            payouts.append(pot_return)
            combo_details.append({
                "legs_indices": list(combo_indices),
                "combo_odds": round(combo_odds, 2),
                "unit_stake": unit_stake,
                "potential_return": pot_return
            })

        # Min payout if exactly k matches win
        min_payout = min(payouts) if payouts else 0.0
        # Max payout if all n matches win (all combos hit)
        max_payout = round(sum(payouts), 2)

        return {
            "system_name": f"{k} of {n}",
            "k": k,
            "n": n,
            "total_combos": total_combos,
            "total_stake": total_stake,
            "unit_stake": unit_stake,
            "min_payout_if_k_win": min_payout,
            "max_payout_if_all_win": max_payout,
            "max_profit": round(max_payout - total_stake, 2),
            "combinations": combo_details[:15]
        }
