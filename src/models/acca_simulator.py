"""
Acca Simulator & Target Odds Multi-Match Optimizer
Simulates and optimizes accumulator combinations matching user-specified
target total odds and match count using Dixon-Coles probabilities,
consensus intelligence, and Expected Value (EV) maximization.
"""

import itertools
import math
import random
from typing import Dict, List, Any, Optional, Tuple


class AccaSimulator:
    """
    Finds mathematically optimal accumulator combinations of N matches
    that achieve an approximated target multiplier (e.g. 5.0x, 10.0x, 20.0x)
    while maximizing joint win probability and positive expected value.
    """

    def __init__(self, fixtures: List[Dict[str, Any]]):
        self.fixtures = fixtures

    def _prepare_candidates(self) -> List[Dict[str, Any]]:
        """Pre-screens matches and derives individual pick odds and win probabilities"""
        candidates = []
        seen_keys = set()

        for f in self.fixtures:
            home = f.get("home_team", "").strip()
            away = f.get("away_team", "").strip()
            if not home or not away:
                continue

            pair_key = f"{home.lower()}_vs_{away.lower()}"
            if pair_key in seen_keys:
                continue
            seen_keys.add(pair_key)

            oh = float(f.get("odds_home", 0.0) or 0.0)
            od = float(f.get("odds_draw", 0.0) or 0.0)
            oa = float(f.get("odds_away", 0.0) or 0.0)

            rec = f.get("recommended_outcome", "Home")
            if rec == "Away" and oa > 1.10:
                pick = "Away"
                pick_odds = oa
            elif rec == "Draw" and od > 1.10:
                pick = "Draw"
                pick_odds = od
            elif oh > 1.10:
                pick = "Home"
                pick_odds = oh
            elif oa > 1.10:
                pick = "Away"
                pick_odds = oa
            else:
                continue

            # Fair win probability estimation
            # 1. Check if Forebet prob exists
            p_fair = 0.0
            if "forebet_probs" in f and f["forebet_probs"]:
                import re
                probs = re.findall(r'[HDA]:(\d+)%', f["forebet_probs"])
                if len(probs) >= 3:
                    if pick == "Home":
                        p_fair = float(probs[0]) / 100.0
                    elif pick == "Draw":
                        p_fair = float(probs[1]) / 100.0
                    else:
                        p_fair = float(probs[2]) / 100.0

            # 2. If no direct external probability, compute vig-free implied probability
            if p_fair <= 0.0:
                inv_h = (1.0 / oh) if oh > 1.0 else 0.4
                inv_d = (1.0 / od) if od > 1.0 else 0.3
                inv_a = (1.0 / oa) if oa > 1.0 else 0.3
                total_inv = inv_h + inv_d + inv_a
                if pick == "Home":
                    p_fair = inv_h / total_inv
                elif pick == "Draw":
                    p_fair = inv_d / total_inv
                else:
                    p_fair = inv_a / total_inv

            # Individual edge
            ev_individual = (p_fair * pick_odds) - 1.0

            candidates.append({
                "fixture": f,
                "home_team": home,
                "away_team": away,
                "league": f.get("league", "Football"),
                "selected_outcome": pick,
                "selected_odds": round(pick_odds, 2),
                "odds_home": oh,
                "odds_draw": od,
                "odds_away": oa,
                "win_prob": round(p_fair, 4),
                "individual_ev": round(ev_individual, 4),
                "sources": f.get("sources", []),
                "sporty_id": f.get("sporty_id", ""),
                "tip_info": f.get("predictz_tip") or f.get("forebet_score") or f.get("betensured_tip") or ""
            })

        return candidates

    def simulate_optimal_slip(
        self,
        target_odds: float = 10.0,
        match_count: int = 5,
        strategy: str = "max_prob",
        max_candidates: int = 40
    ) -> Dict[str, Any]:
        """
        Runs combinatorial search to find the optimal accumulator of length match_count
        that hits approx target_odds.
        """
        raw_candidates = self._prepare_candidates()
        # Prioritize fixtures with verified SportyBet odds
        raw_candidates.sort(key=lambda x: (1 if "SportyBet" in x.get("sources", []) else 0, x["win_prob"]), reverse=True)
        if len(raw_candidates) < match_count:
            return {
                "success": False,
                "message": f"Only {len(raw_candidates)} eligible fixtures available, need at least {match_count}.",
                "candidates": []
            }

        # Filter out extreme longshots unless target is very high
        if target_odds <= 20.0:
            filtered = [c for c in raw_candidates if 1.15 <= c["selected_odds"] <= 5.0]
        else:
            filtered = [c for c in raw_candidates if 1.20 <= c["selected_odds"] <= 8.0]

        if len(filtered) < match_count:
            filtered = raw_candidates

        # Sort filtered candidates according to strategy to prune search space
        if strategy == "max_prob":
            filtered.sort(key=lambda x: (x["win_prob"], -x["selected_odds"]), reverse=True)
        elif strategy == "max_ev":
            filtered.sort(key=lambda x: x["individual_ev"], reverse=True)
        else:  # balanced
            filtered.sort(key=lambda x: (x["win_prob"] * 0.6 + (x["individual_ev"] + 1.0) * 0.4), reverse=True)

        pool = filtered[:30]

        # Target odds tolerance window [target * 0.80, target * 1.30]
        min_target = target_odds * 0.80
        max_target = target_odds * 1.30

        best_slips = []

        # If pool size is reasonable, search exact combinations
        total_combos = math.comb(len(pool), match_count)
        if total_combos <= 8000:
            combo_iter = itertools.combinations(pool, match_count)
        else:
            # Sample 8,000 randomized Monte Carlo combinations using indices for sub-second execution
            def sample_generator():
                seen = set()
                indices = list(range(len(pool)))
                for _ in range(8000):
                    idx_tuple = tuple(sorted(random.sample(indices, match_count)))
                    if idx_tuple not in seen:
                        seen.add(idx_tuple)
                        yield [pool[i] for i in idx_tuple]
            combo_iter = sample_generator()

        for combo in combo_iter:
            # Calculate combined odds
            combined_odds = 1.0
            for item in combo:
                combined_odds *= item["selected_odds"]

            # Filter out combinations wildly outside target range
            diff = abs(combined_odds - target_odds)
            if combined_odds < min_target or combined_odds > max_target:
                # Still record close slips if very few fit
                if diff > target_odds * 0.5:
                    continue

            # Calculate joint probability (independence assumption)
            joint_prob = 1.0
            for item in combo:
                joint_prob *= item["win_prob"]

            ev = (joint_prob * combined_odds) - 1.0

            # Scoring based on strategy and closeness to target
            closeness_penalty = (diff / target_odds) * 0.25

            if strategy == "max_prob":
                score = joint_prob * (1.0 - closeness_penalty)
            elif strategy == "max_ev":
                score = (ev + 1.0) * (1.0 - closeness_penalty)
            else:  # balanced
                score = (joint_prob ** 0.5) * (max(0.1, ev + 1.0) ** 0.5) * (1.0 - closeness_penalty)

            best_slips.append({
                "score": score,
                "combined_odds": round(combined_odds, 2),
                "joint_probability": round(joint_prob * 100, 2),
                "expected_value_pct": round(ev * 100, 2),
                "target_odds": target_odds,
                "matches": list(combo)
            })

        if not best_slips:
            # Fallback: pick the closest combination by odds alone
            combo_fallback = sorted(pool, key=lambda x: x["win_prob"], reverse=True)[:match_count]
            co = 1.0
            jp = 1.0
            for item in combo_fallback:
                co *= item["selected_odds"]
                jp *= item["win_prob"]
            best_slips.append({
                "score": jp,
                "combined_odds": round(co, 2),
                "joint_probability": round(jp * 100, 2),
                "expected_value_pct": round(((jp * co) - 1.0) * 100, 2),
                "target_odds": target_odds,
                "matches": combo_fallback
            })

        # Sort by score descending
        best_slips.sort(key=lambda x: x["score"], reverse=True)

        optimal = best_slips[0]
        runners_up = best_slips[1:3]

        return {
            "success": True,
            "target_odds": target_odds,
            "match_count": match_count,
            "strategy": strategy,
            "optimal_slip": {
                "combined_odds": optimal["combined_odds"],
                "joint_probability": optimal["joint_probability"],
                "expected_value_pct": optimal["expected_value_pct"],
                "payout_10ngn": round(10.0 * optimal["combined_odds"], 2),
                "matches": optimal["matches"]
            },
            "runners_up": [
                {
                    "combined_odds": r["combined_odds"],
                    "joint_probability": r["joint_probability"],
                    "expected_value_pct": r["expected_value_pct"],
                    "payout_10ngn": round(10.0 * r["combined_odds"], 2),
                    "matches": r["matches"]
                }
                for r in runners_up
            ]
        }
