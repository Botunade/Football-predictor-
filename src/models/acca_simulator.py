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

    def _prepare_candidates(self, ideal_leg_odds: float = 1.65, strategy: str = "max_prob", market_filter: str = "all") -> List[Dict[str, Any]]:
        """
        Pre-screens matches and evaluates Multi-Market options (1X2, Double Chance 1X/X2/12,
        Over/Under 1.5/2.5/3.5, BTTS Yes/No, Draw No Bet) to select the best market per match
        tailored to the user's target multiplier and strategy.
        """
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
            if oh <= 1.05 and oa <= 1.05:
                continue

            m_odds = dict(f.get("market_odds") or {})
            inv_h = (1.0 / oh) if oh > 1.0 else 0.40
            inv_d = (1.0 / od) if od > 1.0 else 0.28
            inv_a = (1.0 / oa) if oa > 1.0 else 0.32
            total_inv = inv_h + inv_d + inv_a
            p_h = inv_h / total_inv
            p_d = inv_d / total_inv
            p_a = inv_a / total_inv

            # Blend with Forebet probabilities if available
            if "forebet_probs" in f and f["forebet_probs"]:
                import re
                probs = re.findall(r'[HDA]:(\d+)%', f["forebet_probs"])
                if len(probs) >= 3:
                    p_h = 0.5 * p_h + 0.5 * (float(probs[0]) / 100.0)
                    p_d = 0.5 * p_d + 0.5 * (float(probs[1]) / 100.0)
                    p_a = 0.5 * p_a + 0.5 * (float(probs[2]) / 100.0)

            p_1x = min(0.95, p_h + p_d)
            p_x2 = min(0.95, p_d + p_a)
            p_12 = min(0.95, p_h + p_a)
            p_dnb_h = p_h / max(0.1, p_h + p_a)
            p_dnb_a = p_a / max(0.1, p_h + p_a)
            p_ov15 = min(0.88, 0.72 + 0.15 * abs(p_h - p_a))
            p_ov25 = min(0.75, 0.48 + 0.18 * abs(p_h - p_a))
            p_un35 = min(0.88, 0.74 - 0.10 * abs(p_h - p_a))
            p_btts_y = min(0.72, 0.54 - 0.15 * abs(p_h - p_a))

            def get_m_odd(k: str, p_est: float, fallback: float = 0.0) -> float:
                if k in m_odds and float(m_odds[k]) > 1.04:
                    return float(m_odds[k])
                if fallback > 1.04:
                    return fallback
                return round(max(1.10, 0.93 / max(0.08, p_est)), 2)

            options = [
                ("Home", oh if oh > 1.04 else get_m_odd("Home", p_h), p_h, "1x2"),
                ("Draw", od if od > 1.04 else get_m_odd("Draw", p_d), p_d, "1x2"),
                ("Away", oa if oa > 1.04 else get_m_odd("Away", p_a), p_a, "1x2"),
                ("1X", get_m_odd("1X", p_1x), p_1x, "safe"),
                ("X2", get_m_odd("X2", p_x2), p_x2, "safe"),
                ("12", get_m_odd("12", p_12), p_12, "safe"),
                ("Over 1.5", get_m_odd("Over 1.5", p_ov15), p_ov15, "goals"),
                ("Over 2.5", get_m_odd("Over 2.5", p_ov25), p_ov25, "goals"),
                ("Under 3.5", get_m_odd("Under 3.5", p_un35), p_un35, "goals"),
                ("BTTS Yes", get_m_odd("BTTS Yes", p_btts_y), p_btts_y, "goals"),
                ("Home DNB", get_m_odd("Home DNB", p_dnb_h), p_dnb_h, "safe"),
                ("Away DNB", get_m_odd("Away DNB", p_dnb_a), p_dnb_a, "safe"),
            ]

            if market_filter == "safe":
                filtered_opts = [o for o in options if o[3] in ("safe", "goals") and o[2] >= 0.60]
            elif market_filter == "1x2":
                filtered_opts = [o for o in options if o[3] == "1x2"]
            elif market_filter == "goals":
                filtered_opts = [o for o in options if o[3] == "goals"]
            else:
                filtered_opts = options

            if not filtered_opts:
                filtered_opts = options

            rec_default = f.get("recommended_outcome", "")

            best_opt = None
            best_opt_score = -999.0
            for pick_name, p_odds, p_prob, cat in filtered_opts:
                if p_odds < 1.12 or p_odds > 6.50:
                    continue
                ev_val = (p_prob * p_odds) - 1.0
                dist_penalty = abs(math.log(max(1.1, p_odds)) - math.log(max(1.1, ideal_leg_odds))) * 1.15
                rec_bonus = 0.05 if pick_name == rec_default else 0.0
                if strategy == "max_prob":
                    sc = (p_prob * 1.0) - dist_penalty + rec_bonus
                elif strategy == "max_ev":
                    sc = ev_val - (dist_penalty * 0.8) + (p_prob * 0.4) + rec_bonus
                else:
                    sc = (p_prob * 0.8) + (ev_val * 0.5) - dist_penalty + rec_bonus
                if sc > best_opt_score:
                    best_opt_score = sc
                    best_opt = (pick_name, p_odds, p_prob, ev_val)

            if not best_opt:
                continue

            pick, pick_odds, p_fair, ev_individual = best_opt

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
                "market_odds": m_odds,
                "win_prob": round(p_fair, 4),
                "individual_ev": round(ev_individual, 4),
                "sources": f.get("sources", []),
                "sporty_id": f.get("sporty_id", ""),
                "tip_info": f.get("predictz_tip") or f.get("forebet_score") or f.get("betensured_tip") or f.get("matchoutlook_tip") or f"AI Pick: {pick}"
            })

        return candidates

    def simulate_optimal_slip(
        self,
        target_odds: float = 10.0,
        match_count: int = 5,
        strategy: str = "max_prob",
        stake: float = 10.0,
        market_filter: str = "all",
        max_candidates: int = 40
    ) -> Dict[str, Any]:
        """
        Runs combinatorial search to find the optimal accumulator of length match_count
        that hits approx target_odds across all supported markets.
        """
        ideal_leg = max(1.18, min(3.50, target_odds ** (1.0 / max(1, match_count))))
        raw_candidates = self._prepare_candidates(ideal_leg_odds=ideal_leg, strategy=strategy, market_filter=market_filter)
        # Prioritize fixtures with verified SportyBet odds
        raw_candidates.sort(key=lambda x: (1 if "SportyBet" in x.get("sources", []) else 0, x["win_prob"]), reverse=True)
        if len(raw_candidates) < match_count:
            return {
                "success": False,
                "message": f"Only {len(raw_candidates)} eligible fixtures available, need at least {match_count}.",
                "candidates": []
            }

        # Keep SportyBet-verified fixtures if we have enough of them
        sporty_only = [c for c in raw_candidates if c.get("sporty_id")]
        if len(sporty_only) >= match_count * 2:
            raw_candidates = sporty_only

        # Filter out extreme longshots unless target is very high
        min_leg = max(1.12, ideal_leg * 0.65)
        max_leg = min(8.0, max(3.5, ideal_leg * 1.65))
        filtered = [c for c in raw_candidates if min_leg <= c["selected_odds"] <= max_leg]

        if len(filtered) < max(match_count * 2, 12):
            filtered = [c for c in raw_candidates if 1.15 <= c["selected_odds"] <= 6.0]
        if len(filtered) < match_count:
            filtered = raw_candidates

        def candidate_rank_key(x: Dict[str, Any]) -> Tuple[int, float]:
            has_sp = 1 if x.get("sporty_id") else 0
            dist = abs(math.log(max(1.1, x["selected_odds"])) - math.log(max(1.1, ideal_leg)))
            if strategy == "max_prob":
                sc = (x["win_prob"] * 1.0) - (dist * 0.65)
            elif strategy == "max_ev":
                sc = (x["individual_ev"] * 1.0) - (dist * 0.55)
            else:
                sc = (x["win_prob"] * 0.6 + (x["individual_ev"] + 1.0) * 0.4) - (dist * 0.60)
            return (has_sp, sc)

        filtered.sort(key=candidate_rank_key, reverse=True)
        pool = filtered[:35]

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
            "stake": stake,
            "optimal_slip": {
                "combined_odds": optimal["combined_odds"],
                "joint_probability": optimal["joint_probability"],
                "expected_value_pct": optimal["expected_value_pct"],
                "payout_10ngn": round(stake * optimal["combined_odds"], 2),
                "payout_stake": round(stake * optimal["combined_odds"], 2),
                "matches": optimal["matches"]
            },
            "runners_up": [
                {
                    "combined_odds": r["combined_odds"],
                    "joint_probability": r["joint_probability"],
                    "expected_value_pct": r["expected_value_pct"],
                    "payout_10ngn": round(stake * r["combined_odds"], 2),
                    "payout_stake": round(stake * r["combined_odds"], 2),
                    "matches": r["matches"]
                }
                for r in runners_up
            ]
        }
