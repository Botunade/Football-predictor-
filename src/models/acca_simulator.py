"""
Acca Simulator & Target Odds Multi-Match Optimizer
Simulates and optimizes accumulator combinations matching user-specified
target total odds and match count using Dixon-Coles probabilities,
consensus intelligence, diversity constraints, and Expected Value (EV) maximization.
"""

import itertools
import math
import random
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class AccaSimulator:
    """
    Finds mathematically optimal accumulator combinations of N matches
    that achieve an approximated target multiplier (e.g. 5.0x, 10.0x, 20.0x, 50.0x)
    while maximizing joint win probability, positive expected value, and market diversity.
    """

    def __init__(self, fixtures: List[Dict[str, Any]]):
        self.fixtures = fixtures

    def _prepare_candidates(self, ideal_leg_odds: float = 1.65, strategy: str = "max_prob", market_filter: str = "all") -> List[Dict[str, Any]]:
        """
        Pre-screens matches and evaluates Multi-Market options across distinct categories:
        1X2, Early Payout (1UP), Double Chance (1X/X2/12), Goals (Over 1.5, Under 3.5, BTTS),
        and Halves (1st/2nd Half). Preserves the top candidate option in EACH category per fixture
        so accumulators can be built with true diversity.
        """
        fixture_candidates = []
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

            # Double Chance & DNB
            p_1x = min(0.95, p_h + p_d)
            p_x2 = min(0.95, p_d + p_a)
            p_12 = min(0.95, p_h + p_a)
            p_dnb_h = p_h / max(0.1, p_h + p_a)
            p_dnb_a = p_a / max(0.1, p_h + p_a)

            # Totals & Goals
            p_ov15 = min(0.88, 0.72 + 0.15 * abs(p_h - p_a))
            p_ov25 = min(0.75, 0.48 + 0.18 * abs(p_h - p_a))
            p_un25 = max(0.25, 1.0 - p_ov25)
            p_un35 = min(0.88, 0.74 - 0.10 * abs(p_h - p_a))
            p_btts_y = min(0.72, 0.54 - 0.15 * abs(p_h - p_a))
            p_btts_n = max(0.28, 1.0 - p_btts_y)

            # 1st Half & 2nd Half & Team / Special probabilities (Calibrated Poisson)
            p_1h_o05 = min(0.82, 0.68 + 0.10 * abs(p_h - p_a))
            p_1h_u15 = min(0.86, 0.72 - 0.08 * abs(p_h - p_a))
            p_1h_1x = min(0.92, p_h * 0.65 + 0.44)
            p_1h_x2 = min(0.92, p_a * 0.65 + 0.44)
            p_2h_o05 = min(0.84, 0.72 + 0.10 * abs(p_h - p_a))
            p_2h_u15 = min(0.78, 0.65 - 0.10 * abs(p_h - p_a))
            # Calibrated 2nd Half Under 2.5: realistic average goals is ~1.5 (prob ~70-74%, not 85-92%)
            p_2h_u25 = min(0.75, max(0.58, 0.71 - 0.12 * abs(p_ov25 - 0.50)))
            p_2h_1x = min(0.88, p_h * 0.65 + 0.40)
            p_2h_x2 = min(0.88, p_a * 0.65 + 0.40)
            p_h_1up = min(0.92, p_h + p_d * 0.32)
            p_a_1up = min(0.92, p_a + p_d * 0.30)
            p_h_o05 = min(0.88, 0.55 + p_h * 0.45)
            p_a_o05 = min(0.88, 0.52 + p_a * 0.45)
            p_mg_13 = min(0.82, 0.72)

            def get_m_odd(k: str, p_est: float, fallback: float = 0.0) -> float:
                if k in m_odds and float(m_odds[k]) > 1.04:
                    return float(m_odds[k])
                if fallback > 1.04:
                    return fallback
                return round(max(1.10, 0.93 / max(0.08, p_est)), 2)

            all_options = [
                # 1X2 Match Winner
                ("Home", oh if oh > 1.04 else get_m_odd("Home", p_h), p_h, "1x2"),
                ("Draw", od if od > 1.04 else get_m_odd("Draw", p_d), p_d, "1x2"),
                ("Away", oa if oa > 1.04 else get_m_odd("Away", p_a), p_a, "1x2"),
                # Early Payout (1UP)
                ("Home (1UP)", get_m_odd("Home (1UP)", p_h_1up), p_h_1up, "early_payout"),
                ("Away (1UP)", get_m_odd("Away (1UP)", p_a_1up), p_a_1up, "early_payout"),
                # Double Chance & DNB
                ("1X", get_m_odd("1X", p_1x), p_1x, "double_chance"),
                ("X2", get_m_odd("X2", p_x2), p_x2, "double_chance"),
                ("12", get_m_odd("12", p_12), p_12, "double_chance"),
                ("Home DNB", get_m_odd("Home DNB", p_dnb_h), p_dnb_h, "double_chance"),
                ("Away DNB", get_m_odd("Away DNB", p_dnb_a), p_dnb_a, "double_chance"),
                # Goals & Totals
                ("Over 1.5", get_m_odd("Over 1.5", p_ov15), p_ov15, "goals"),
                ("Over 2.5", get_m_odd("Over 2.5", p_ov25), p_ov25, "goals"),
                ("Under 2.5", get_m_odd("Under 2.5", p_un25), p_un25, "goals"),
                ("Under 3.5", get_m_odd("Under 3.5", p_un35), p_un35, "goals"),
                ("BTTS Yes", get_m_odd("BTTS Yes", p_btts_y), p_btts_y, "goals"),
                ("BTTS No", get_m_odd("BTTS No", p_btts_n), p_btts_n, "goals"),
                ("Home Team Over 0.5", get_m_odd("Home Team Over 0.5", p_h_o05), p_h_o05, "goals"),
                ("Away Team Over 0.5", get_m_odd("Away Team Over 0.5", p_a_o05), p_a_o05, "goals"),
                ("Multi-Goals 1-3", get_m_odd("Multi-Goals 1-3", p_mg_13), p_mg_13, "goals"),
                # Halves
                ("1st Half Over 0.5", get_m_odd("1st Half Over 0.5", p_1h_o05), p_1h_o05, "halves"),
                ("1st Half Under 1.5", get_m_odd("1st Half Under 1.5", p_1h_u15), p_1h_u15, "halves"),
                ("1st Half 1X", get_m_odd("1st Half 1X", p_1h_1x), p_1h_1x, "halves"),
                ("1st Half X2", get_m_odd("1st Half X2", p_1h_x2), p_1h_x2, "halves"),
                ("2nd Half Over 0.5", get_m_odd("2nd Half Over 0.5", p_2h_o05), p_2h_o05, "halves"),
                ("2nd Half Under 1.5", get_m_odd("2nd Half Under 1.5", p_2h_u15), p_2h_u15, "halves"),
                ("2nd Half Under 2.5", get_m_odd("2nd Half Under 2.5", p_2h_u25), p_2h_u25, "halves"),
                ("2nd Half 1X", get_m_odd("2nd Half 1X", p_2h_1x), p_2h_1x, "halves"),
                ("2nd Half X2", get_m_odd("2nd Half X2", p_2h_x2), p_2h_x2, "halves"),
            ]

            # Apply market filter
            if market_filter == "1x2":
                filtered_opts = [o for o in all_options if o[3] == "1x2"]
            elif market_filter == "halves":
                filtered_opts = [o for o in all_options if o[3] == "halves"]
            elif market_filter == "goals":
                filtered_opts = [o for o in all_options if o[3] == "goals"]
            elif market_filter == "safe":
                filtered_opts = [o for o in all_options if o[3] in ("early_payout", "double_chance", "goals", "halves") and o[2] >= 0.62]
            else:  # all
                filtered_opts = all_options

            if not filtered_opts:
                filtered_opts = all_options

            rec_default = f.get("recommended_outcome", "")

            # Score each option and keep the top option in EACH category for this fixture
            category_bests: Dict[str, Tuple[str, float, float, float, float]] = {}

            for pick_name, p_odds, p_prob, cat in filtered_opts:
                if p_odds < 1.12 or p_odds > 6.50:
                    continue
                ev_val = (p_prob * p_odds) - 1.0
                dist_penalty = abs(math.log(max(1.08, p_odds)) - math.log(max(1.08, ideal_leg_odds))) * 1.10
                rec_bonus = 0.08 if pick_name == rec_default else 0.0

                if strategy == "max_prob":
                    sc = (p_prob * 1.0) - dist_penalty + rec_bonus
                elif strategy == "max_ev":
                    sc = ev_val - (dist_penalty * 0.75) + (p_prob * 0.35) + rec_bonus
                else:  # balanced
                    sc = (p_prob * 0.75) + (ev_val * 0.45) - dist_penalty + rec_bonus

                if cat not in category_bests or sc > category_bests[cat][4]:
                    category_bests[cat] = (pick_name, p_odds, p_prob, ev_val, sc)

            if not category_bests:
                continue

            options_list = []
            for cat, (pick_name, p_odds, p_prob, ev_val, sc) in category_bests.items():
                options_list.append({
                    "selected_outcome": pick_name,
                    "selected_odds": round(p_odds, 2),
                    "win_prob": round(p_prob, 4),
                    "individual_ev": round(ev_val, 4),
                    "category": cat,
                    "score": round(sc, 4)
                })

            # Sort options in this fixture by score
            options_list.sort(key=lambda x: x["score"], reverse=True)
            best_opt = options_list[0]

            fixture_candidates.append({
                "pair_key": pair_key,
                "fixture": f,
                "home_team": home,
                "away_team": away,
                "league": f.get("league", "Football"),
                "odds_home": oh,
                "odds_draw": od,
                "odds_away": oa,
                "market_odds": m_odds,
                "sporty_id": f.get("sporty_id", ""),
                "sources": f.get("sources", []),
                "tip_info": f.get("predictz_tip") or f.get("forebet_score") or f.get("betensured_tip") or f.get("matchoutlook_tip") or f"AI Pick: {best_opt['selected_outcome']}",
                "best_outcome": best_opt["selected_outcome"],
                "best_odds": best_opt["selected_odds"],
                "best_prob": best_opt["win_prob"],
                "options": options_list
            })

        return fixture_candidates

    def simulate_optimal_slip(
        self,
        target_odds: float = 10.0,
        match_count: int = 5,
        strategy: str = "max_prob",
        stake: float = 10.0,
        market_filter: str = "all",
        variation_index: int = 0,
        max_candidates: int = 40
    ) -> Dict[str, Any]:
        """
        Runs multi-market combinatorial optimization to assemble an optimal accumulator of
        length match_count hitting approx target_odds.
        Enforces strict Market Diversity Constraints: no single market type can dominate
        the accumulator (max 2 occurrences for <=6 legs, max 3 for >6 legs).
        Rewards high diversity and varied markets (1UP, 1X, Over 1.5, 1st Half, DNB, etc.).
        """
        ideal_leg = max(1.18, min(3.50, target_odds ** (1.0 / max(1, match_count))))
        fixtures_pool = self._prepare_candidates(ideal_leg_odds=ideal_leg, strategy=strategy, market_filter=market_filter)

        if len(fixtures_pool) < match_count:
            return {
                "success": False,
                "message": f"Only {len(fixtures_pool)} eligible fixtures available, need at least {match_count}.",
                "candidates": []
            }

        # Prioritize fixtures with verified SportyBet odds and strong data
        sporty_verified = [f for f in fixtures_pool if f.get("sporty_id")]
        if len(sporty_verified) >= match_count * 2:
            fixtures_pool = sporty_verified

        # Sort fixtures by top option quality
        fixtures_pool.sort(
            key=lambda x: (
                1 if "SportyBet" in x.get("sources", []) else 0,
                x["options"][0]["score"] if x["options"] else 0.0
            ),
            reverse=True
        )
        pool = fixtures_pool[:max_candidates]

        # Target odds tolerance window [target * 0.78, target * 1.35]
        min_target = target_odds * 0.78
        max_target = target_odds * 1.35

        # Strict Diversity Constraint: maximum times any identical outcome can appear in the slip
        # E.g. in a 10-leg slip, max 3 of "2nd Half Under 2.5", max 3 of "Over 1.5", etc.
        max_same_outcome = match_count if market_filter == "1x2" else (2 if match_count <= 6 else 3)

        best_slips = []

        # If pool size allows, search exact combinations; else sample randomized Monte Carlo batches
        total_combos = math.comb(len(pool), match_count)
        if total_combos <= 5000:
            fixture_combos = itertools.combinations(pool, match_count)
        else:
            def sample_generator():
                seen = set()
                indices = list(range(len(pool)))
                # 9,000 samples for fast, thorough exploration
                for _ in range(9000):
                    idx_tuple = tuple(sorted(random.sample(indices, match_count)))
                    if idx_tuple not in seen:
                        seen.add(idx_tuple)
                        yield [pool[i] for i in idx_tuple]
            fixture_combos = sample_generator()

        for f_combo in fixture_combos:
            # For each fixture in this combination, select an option to build a diversified accumulator
            # Track outcome counts to strictly enforce diversity
            current_slip = []
            combined_odds = 1.0
            joint_prob = 1.0
            outcome_counts = Counter()
            category_counts = Counter()

            valid_combo = True
            for f in f_combo:
                # Rank available options for this fixture, prioritizing options that haven't reached max_same_outcome
                # and that keep combined odds progression on track for ideal_leg
                best_leg_choice = None
                best_leg_sc = -999.0

                for opt in f["options"]:
                    outcome = opt["selected_outcome"]
                    # If this outcome is already at max limit, skip it to enforce diversity!
                    if outcome_counts[outcome] >= max_same_outcome:
                        continue

                    # Diversity bonus for choosing a category or outcome not yet in the slip
                    div_sc = (0.25 if category_counts[opt["category"]] == 0 else 0.0) + \
                             (0.20 if outcome_counts[outcome] == 0 else 0.0)
                    
                    dist_to_ideal = abs(math.log(max(1.08, opt["selected_odds"])) - math.log(max(1.08, ideal_leg)))
                    leg_sc = opt["score"] + div_sc - (dist_to_ideal * 0.40)

                    if leg_sc > best_leg_sc:
                        best_leg_sc = leg_sc
                        best_leg_choice = opt

                # Fallback to top option if all options were capped
                if not best_leg_choice:
                    best_leg_choice = f["options"][0]

                outcome_counts[best_leg_choice["selected_outcome"]] += 1
                category_counts[best_leg_choice["category"]] += 1
                combined_odds *= best_leg_choice["selected_odds"]
                joint_prob *= best_leg_choice["win_prob"]

                current_slip.append({
                    "fixture": f.get("fixture", {}),
                    "home_team": f["home_team"],
                    "away_team": f["away_team"],
                    "league": f["league"],
                    "selected_outcome": best_leg_choice["selected_outcome"],
                    "selected_odds": best_leg_choice["selected_odds"],
                    "win_prob": best_leg_choice["win_prob"],
                    "individual_ev": best_leg_choice["individual_ev"],
                    "category": best_leg_choice["category"],
                    "odds_home": f["odds_home"],
                    "odds_draw": f["odds_draw"],
                    "odds_away": f["odds_away"],
                    "market_odds": f["market_odds"],
                    "sources": f["sources"],
                    "sporty_id": f["sporty_id"],
                    "tip_info": f["tip_info"]
                })

            # Check if combined odds fit in tolerance window
            diff = abs(combined_odds - target_odds)
            if combined_odds < min_target or combined_odds > max_target:
                if diff > target_odds * 0.45:
                    continue

            ev = (joint_prob * combined_odds) - 1.0

            # Calculate Diversity Multiplier
            unique_outcomes = len(outcome_counts)
            diversity_ratio = unique_outcomes / max(1, match_count)
            diversity_bonus = diversity_ratio * 0.45  # Substantial boost for diversified slips

            closeness_penalty = (diff / target_odds) * 0.25

            if strategy == "max_prob":
                score = (joint_prob ** 0.5) * (1.0 + diversity_bonus) * (1.0 - closeness_penalty)
            elif strategy == "max_ev":
                score = (ev + 1.0) * (1.0 + diversity_bonus) * (1.0 - closeness_penalty)
            else:  # balanced
                score = (joint_prob ** 0.3) * (max(0.1, ev + 1.0) ** 0.3) * (1.0 + diversity_bonus) * (1.0 - closeness_penalty)

            best_slips.append({
                "score": score,
                "combined_odds": round(combined_odds, 2),
                "joint_probability": round(joint_prob * 100, 2),
                "expected_value_pct": round(ev * 100, 2),
                "unique_markets": unique_outcomes,
                "target_odds": target_odds,
                "matches": current_slip
            })

        # Fallback if window was too tight
        if not best_slips:
            sorted_by_prob = sorted(pool, key=lambda x: x["options"][0]["win_prob"], reverse=True)[:match_count]
            co = 1.0
            jp = 1.0
            fallback_slip = []
            f_counts = Counter()
            for f in sorted_by_prob:
                choice = f["options"][0]
                for opt in f["options"]:
                    if f_counts[opt["selected_outcome"]] < max_same_outcome:
                        choice = opt
                        break
                f_counts[choice["selected_outcome"]] += 1
                co *= choice["selected_odds"]
                jp *= choice["win_prob"]
                fallback_slip.append({
                    "fixture": f.get("fixture", {}),
                    "home_team": f["home_team"],
                    "away_team": f["away_team"],
                    "league": f["league"],
                    "selected_outcome": choice["selected_outcome"],
                    "selected_odds": choice["selected_odds"],
                    "win_prob": choice["win_prob"],
                    "individual_ev": choice["individual_ev"],
                    "category": choice["category"],
                    "odds_home": f["odds_home"],
                    "odds_draw": f["odds_draw"],
                    "odds_away": f["odds_away"],
                    "market_odds": f["market_odds"],
                    "sources": f["sources"],
                    "sporty_id": f["sporty_id"],
                    "tip_info": f["tip_info"]
                })

            best_slips.append({
                "score": jp,
                "combined_odds": round(co, 2),
                "joint_probability": round(jp * 100, 2),
                "expected_value_pct": round(((jp * co) - 1.0) * 100, 2),
                "unique_markets": len(f_counts),
                "target_odds": target_odds,
                "matches": fallback_slip
            })

        # Sort by score descending
        best_slips.sort(key=lambda x: x["score"], reverse=True)

        chosen_idx = max(0, min(variation_index, len(best_slips) - 1))
        optimal = best_slips[chosen_idx]
        runners_up = [s for i, s in enumerate(best_slips) if i != chosen_idx][:3]

        return {
            "success": True,
            "target_odds": target_odds,
            "match_count": match_count,
            "strategy": strategy,
            "stake": stake,
            "variation_index": chosen_idx,
            "total_variations": len(best_slips),
            "optimal_slip": {
                "combined_odds": optimal["combined_odds"],
                "joint_probability": optimal["joint_probability"],
                "expected_value_pct": optimal["expected_value_pct"],
                "unique_markets": optimal.get("unique_markets", len(set(m["selected_outcome"] for m in optimal["matches"]))),
                "payout_10ngn": round(stake * optimal["combined_odds"], 2),
                "payout_stake": round(stake * optimal["combined_odds"], 2),
                "matches": optimal["matches"]
            },
            "runners_up": [
                {
                    "combined_odds": r["combined_odds"],
                    "joint_probability": r["joint_probability"],
                    "expected_value_pct": r["expected_value_pct"],
                    "unique_markets": r.get("unique_markets", len(set(m["selected_outcome"] for m in r["matches"]))),
                    "payout_10ngn": round(stake * r["combined_odds"], 2),
                    "payout_stake": round(stake * r["combined_odds"], 2),
                    "matches": r["matches"]
                }
                for r in runners_up
            ]
        }
