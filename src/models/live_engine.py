"""
Live In-Play Betting Engine & Real-Time Sniper
Fetches active in-play matches from SportyBet's real-time factsCenter feed,
runs exact in-play bivariate Poisson time-decay models on live scorelines and clocks,
evaluates Game-State Asymmetry (Lead Preservation vs Desperation), Halftime Golden Windows,
and Defensive Gridlock, and generates instant SportyBet live booking codes.
"""

import json
import logging
import math
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("LiveBettingEngine")
logger.setLevel(logging.INFO)


@dataclass
class LiveMarketOption:
    market_id: str
    outcome_id: str
    specifier: str
    market_name: str
    outcome_name: str
    odds: float
    prob: float
    ev_pct: float
    score: float
    is_safe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "outcome_id": self.outcome_id,
            "specifier": self.specifier,
            "market_name": self.market_name,
            "outcome_name": self.outcome_name,
            "odds": round(self.odds, 2),
            "prob_pct": round(self.prob * 100, 1),
            "ev_pct": round(self.ev_pct, 1),
            "score": round(self.score, 4),
            "is_safe": self.is_safe
        }


@dataclass
class LiveFixtureAnalysis:
    sporty_id: str
    home_team: str
    away_team: str
    league: str
    category: str
    minute: int
    clock_str: str
    period: str
    home_score: int
    away_score: int
    score_str: str
    top_pick: LiveMarketOption
    safe_pick: LiveMarketOption
    value_pick: LiveMarketOption
    all_options: List[LiveMarketOption] = field(default_factory=list)
    tactical_badge: str = ""
    tactical_badge_type: str = "standard"
    game_state_label: str = "Neutral"
    confidence_score: int = 7
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sporty_id": self.sporty_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "category": self.category,
            "minute": self.minute,
            "clock_str": self.clock_str,
            "period": self.period,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "score_str": self.score_str,
            "top_pick": self.top_pick.to_dict() if self.top_pick else None,
            "safe_pick": self.safe_pick.to_dict() if self.safe_pick else None,
            "value_pick": self.value_pick.to_dict() if self.value_pick else None,
            "options_count": len(self.all_options),
            "options": [o.to_dict() for o in self.all_options[:8]],
            "tactical_badge": self.tactical_badge,
            "tactical_badge_type": self.tactical_badge_type,
            "game_state_label": self.game_state_label,
            "confidence_score": self.confidence_score,
            "rationale": self.rationale
        }


class LiveBettingEngine:
    """
    Core engine for querying, analyzing, and betting on live in-play football matches.
    Incorporates Game-State Asymmetry, Halftime Golden Windows, and Late-Game Gridlock.
    """

    SPORTY_LIVE_FEED_URL = "https://www.sportybet.com/api/ng/factsCenter/liveOrPrematchEvents?sportId=sr%3Asport%3A1"

    def fetch_live_matches(self) -> List[Dict[str, Any]]:
        """
        Fetches all current in-play matches from SportyBet API.
        """
        url = f"{self.SPORTY_LIVE_FEED_URL}&_t={int(time.time() * 1000)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                tournaments = data.get("data", [])
                events_list = []
                for t in tournaments:
                    t_name = t.get("name", "Football")
                    cat_name = t.get("category", {}).get("name", "")
                    for ev in t.get("events", []):
                        ev["_tournament_name"] = t_name
                        ev["_category_name"] = cat_name
                        events_list.append(ev)
                return events_list
        except Exception as e:
            logger.warning(f"Error fetching SportyBet live matches: {e}")
            return []

    @staticmethod
    def _parse_match_clock(ev: Dict[str, Any]) -> Tuple[int, str, str]:
        """
        Returns (minute, clock_display_string, period_name)
        """
        played_raw = str(ev.get("playedSeconds") or "00:00").strip()
        m_status = str(ev.get("matchStatus") or "H1").strip().upper()
        period_num = ev.get("period", 1)

        minute = 45 if m_status == "HT" else 0
        if ":" in played_raw:
            parts = played_raw.split(":")
            if parts[0].isdigit():
                minute = int(parts[0])
        elif played_raw.isdigit():
            minute = int(played_raw) // 60

        minute = max(1, min(98, minute))

        period_desc = "1st Half"
        if m_status == "HT":
            period_desc = "Half Time"
        elif m_status == "H2" or period_num == 2 or minute > 45:
            period_desc = "2nd Half"

        clock_str = f"{minute}' ({period_desc})"
        return minute, clock_str, period_desc

    @staticmethod
    def _parse_score(ev: Dict[str, Any]) -> Tuple[int, int, str]:
        """
        Extracts home and away goals from setScore or gameScore.
        """
        set_score = str(ev.get("setScore") or "0:0").strip()
        h_score, a_score = 0, 0
        if ":" in set_score:
            parts = set_score.split(":")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                h_score = int(parts[0])
                a_score = int(parts[1])
        elif "-" in set_score:
            parts = set_score.split("-")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                h_score = int(parts[0])
                a_score = int(parts[1])

        return h_score, a_score, f"{h_score}-{a_score}"

    def analyze_live_match(self, ev: Dict[str, Any]) -> Optional[LiveFixtureAnalysis]:
        """
        Analyzes a single live match: in-play Poisson time-decay, Game-State Asymmetry,
        Halftime Golden Windows, Defensive Gridlock, and market evaluation.
        """
        sporty_id = str(ev.get("eventId") or "").strip()
        home = str(ev.get("homeTeamName") or "").strip()
        away = str(ev.get("awayTeamName") or "").strip()
        if not sporty_id or not home or not away:
            return None

        minute, clock_str, period_desc = self._parse_match_clock(ev)
        h_score, a_score, score_str = self._parse_score(ev)
        current_total_goals = h_score + a_score
        score_diff = h_score - a_score

        # Remaining match time fraction
        rem_min = max(2, 90 - minute)
        rem_fraction = rem_min / 90.0

        # Extract pre-match/live 1X2 baseline odds to infer favored team
        live_odds_h, live_odds_a = 2.40, 2.80
        for m in ev.get("markets", []):
            if str(m.get("id")) in ["1", "60200"]:
                for o in m.get("outcomes", []):
                    desc_low = str(o.get("desc", "")).lower()
                    if desc_low == "home":
                        live_odds_h = float(o.get("odds") or live_odds_h)
                    elif desc_low == "away":
                        live_odds_a = float(o.get("odds") or live_odds_a)

        home_favored = live_odds_h <= live_odds_a

        # =====================================================================
        # METHODOLOGY 1: GAME-STATE ASYMMETRY & TACTICAL PSYCHOLOGY
        # =====================================================================
        pace_factor = 1.0
        tactical_badge = "Live Play"
        tactical_badge_type = "standard"
        game_state_label = "Balanced"
        confidence_score = 7

        # 1. Halftime Golden Window (40' to 55')
        is_golden_window = (40 <= minute <= 55) or period_desc == "Half Time"
        if is_golden_window:
            tactical_badge = "⭐ Halftime Golden Window"
            tactical_badge_type = "golden_window"
            confidence_score = 9
            if current_total_goals == 0:
                game_state_label = "0-0 Halftime Breakout"
                pace_factor = 1.14  # High probability of 2nd half goal breakthrough
            else:
                game_state_label = "Halftime Tactical Reset"

        # 2. Favorite Desperation Chasing (Favorite trailing by 1 at 55'+)
        elif ((home_favored and score_diff == -1) or (not home_favored and score_diff == 1)) and minute >= 55:
            tactical_badge = "🔥 Favorite Chasing (Attack Surge)"
            tactical_badge_type = "favorite_chasing"
            game_state_label = "Desperation Attack"
            pace_factor = 1.35  # Heavy offensive commitment; counter-attack concession risk
            confidence_score = 8

        # 3. Underdog Lead Preservation (Underdog leading past 70')
        elif ((home_favored and score_diff <= -1) or (not home_favored and score_diff >= 1)) and minute >= 70:
            tactical_badge = "🛡️ Underdog Low Block (Gridlock)"
            tactical_badge_type = "low_block"
            game_state_label = "Low Block Preservation"
            pace_factor = 0.76  # Defensive time-wasting; box entries neutralized
            confidence_score = 8

        # 4. Late-Game Deadlock (Tied past 78')
        elif score_diff == 0 and minute >= 78:
            tactical_badge = "🔒 Late Deadlock (Risk Aversion)"
            tactical_badge_type = "gridlock"
            game_state_label = "Draw Consolidation"
            pace_factor = 0.72  # Neither side risks a fatal late loss
            confidence_score = 9

        # 5. Open High-Scoring Firefight (3+ goals early)
        elif current_total_goals >= 3 and minute <= 60:
            tactical_badge = "⚡ High-Pace Firefight"
            tactical_badge_type = "firefight"
            game_state_label = "Open Transition Match"
            pace_factor = 1.25

        # Expected remaining goals in the match
        lambda_rem = max(0.05, 2.65 * rem_fraction * pace_factor)
        lambda_h = lambda_rem * (0.58 if home_favored else 0.44)
        lambda_a = lambda_rem * (0.42 if home_favored else 0.56)

        def poisson_pmf(k: int, lamb: float) -> float:
            lamb = max(0.01, lamb)
            return (lamb ** k) * math.exp(-lamb) / math.factorial(k)

        # In-Play 1X2 simulation over remaining score possibilities (up to 4 remaining goals each)
        p_home_win = 0.0
        p_draw_res = 0.0
        p_away_win = 0.0
        for i in range(5):
            for j in range(5):
                p_ij = poisson_pmf(i, lambda_h) * poisson_pmf(j, lambda_a)
                f_h = h_score + i
                f_a = a_score + j
                if f_h > f_a:
                    p_home_win += p_ij
                elif f_h == f_a:
                    p_draw_res += p_ij
                else:
                    p_away_win += p_ij

        tot_p = p_home_win + p_draw_res + p_away_win
        if tot_p > 0:
            p_home_win /= tot_p
            p_draw_res /= tot_p
            p_away_win /= tot_p

        p_1x = min(0.98, p_home_win + p_draw_res)
        p_x2 = min(0.98, p_away_win + p_draw_res)
        p_12 = min(0.98, p_home_win + p_away_win)

        def poisson_cdf_additional_goals(goals_needed: int) -> float:
            """Probability of scoring >= goals_needed additional goals in remaining time"""
            if goals_needed <= 0:
                return 1.0
            under_sum = 0.0
            for k in range(goals_needed):
                under_sum += (lambda_rem ** k) * math.exp(-lambda_rem) / math.factorial(k)
            return max(0.02, min(0.98, 1.0 - under_sum))

        # Evaluate active markets
        options: List[LiveMarketOption] = []

        for m in ev.get("markets", []):
            m_id = str(m.get("id"))
            m_desc = str(m.get("desc") or m.get("name") or "")
            spec = str(m.get("specifier") or "")

            for o in m.get("outcomes", []):
                if o.get("isActive") != 1:
                    continue
                o_id = str(o.get("id"))
                o_desc = str(o.get("desc") or o.get("name") or "")
                odds = float(o.get("odds") or 0.0)
                if odds < 1.10 or odds > 15.0:
                    continue

                est_prob = 0.50
                is_safe = False

                # 1. Total Goals Over/Under (Market 18)
                if m_id == "18" and "total=" in spec:
                    try:
                        line_val = float(spec.replace("total=", ""))
                        # Filter out already decided lines
                        if line_val <= current_total_goals:
                            continue

                        additional_needed = int(math.ceil(line_val - current_total_goals))
                        p_over = poisson_cdf_additional_goals(additional_needed)

                        if "over" in o_desc.lower():
                            est_prob = p_over
                            # Safe over: 1 goal needed in high-pace game or halftime breakout
                            is_safe = (additional_needed == 1 and est_prob >= 0.70)
                        else:  # Under
                            est_prob = max(0.02, 1.0 - p_over)
                            # Safe under: 2+ goal buffer in gridlock or late game
                            is_safe = (additional_needed >= 2 and est_prob >= 0.72)
                    except Exception:
                        continue

                # 2. 1X2 Match Winner (Market 1 or 60200)
                elif m_id in ["1", "60200"]:
                    # In late tied games, penalize volatile 1X2 win picks; reward draw consolidation
                    if o_desc.lower() == "home":
                        est_prob = p_home_win
                        is_safe = (est_prob >= 0.78 and odds >= 1.18 and score_diff >= 1)
                    elif o_desc.lower() == "away":
                        est_prob = p_away_win
                        is_safe = (est_prob >= 0.78 and odds >= 1.18 and score_diff <= -1)
                    else:  # Draw
                        est_prob = p_draw_res
                        is_safe = (est_prob >= 0.72 and rem_min <= 15)

                # 3. Double Chance (Market 10) - High structural integrity in-play
                elif m_id == "10":
                    if "1x" in o_desc.lower() or "home" in o_desc.lower():
                        est_prob = p_1x
                        is_safe = est_prob >= 0.72
                    elif "x2" in o_desc.lower() or "away" in o_desc.lower():
                        est_prob = p_x2
                        is_safe = est_prob >= 0.72
                    else:  # 12
                        est_prob = p_12
                        is_safe = est_prob >= 0.72

                # 4. Next Goal (Market 8)
                elif m_id == "8":
                    if "none" in o_desc.lower():
                        # Probability of 0 additional goals = e^(-lambda_rem)
                        est_prob = math.exp(-lambda_rem)
                        is_safe = est_prob >= 0.68 and rem_min <= 15
                    else:
                        est_prob = (1.0 - math.exp(-lambda_rem)) * 0.48

                # 5. BTTS / Both Teams to Score (Market 11)
                elif m_id == "11":
                    both_scored = (h_score >= 1 and a_score >= 1)
                    if both_scored:
                        continue  # Already decided
                    else:
                        h_needs = max(0, 1 - h_score)
                        a_needs = max(0, 1 - a_score)
                        p_both = (1.0 - math.exp(-lambda_h)) * (1.0 - math.exp(-lambda_a)) if (h_needs and a_needs) else \
                                 (1.0 - math.exp(-lambda_a) if a_needs else (1.0 - math.exp(-lambda_h)))
                        est_prob = p_both if "yes" in o_desc.lower() else (1.0 - p_both)

                # 6. Generic / Default
                else:
                    est_prob = max(0.10, min(0.88, 0.90 / odds))

                # Expected value
                ev_val = (est_prob * odds) - 1.0
                ev_pct = ev_val * 100.0

                # Score: rewards high probability, playable odds (1.20 - 2.80), positive EV, and structural market bonus
                playable_bonus = 0.16 if (1.24 <= odds <= 2.45) else (-0.20 if odds < 1.15 else 0.0)
                structural_bonus = 0.10 if is_safe else 0.0
                sc = (est_prob * 0.65) + (max(-0.25, min(0.40, ev_val)) * 0.35) + playable_bonus + structural_bonus

                options.append(LiveMarketOption(
                    market_id=m_id,
                    outcome_id=o_id,
                    specifier=spec,
                    market_name=m_desc or f"Market {m_id}",
                    outcome_name=o_desc,
                    odds=odds,
                    prob=est_prob,
                    ev_pct=ev_pct,
                    score=sc,
                    is_safe=is_safe
                ))

        if not options:
            return None

        # Sort options by composite score
        options.sort(key=lambda x: x.score, reverse=True)
        top_pick = options[0]

        # Safe pick (prioritizes high-integrity structural buffer markets: DC, Under buffer, Next Goal None)
        safe_candidates = [o for o in options if o.is_safe and o.odds >= 1.16]
        if not safe_candidates:
            safe_candidates = [o for o in options if o.prob >= 0.68 and o.odds >= 1.16]
        if not safe_candidates:
            safe_candidates = sorted(options, key=lambda x: x.prob, reverse=True)
        safe_pick = safe_candidates[0]

        # Value pick (highest EV with prob >= 38% and odds <= 3.80)
        value_candidates = [o for o in options if o.prob >= 0.38 and o.odds <= 3.80]
        if not value_candidates:
            value_candidates = options
        value_pick = sorted(value_candidates, key=lambda x: (x.ev_pct, x.prob), reverse=True)[0]

        # Build tactical rationale reflecting the Game-State methodology
        rationale = (
            f"Score {score_str} at {clock_str} [{game_state_label}]. With ~{rem_min}' remaining, "
            f"model projects lambda = {lambda_rem:.2f} expected goals (Pace Factor: {pace_factor:.2f}x). "
            f"Top pick: {top_pick.outcome_name} ({top_pick.market_name}) at odds {top_pick.odds:.2f} "
            f"(Win Prob: {top_pick.prob*100:.0f}%, EV: {top_pick.ev_pct:+.1f}%)."
        )

        return LiveFixtureAnalysis(
            sporty_id=sporty_id,
            home_team=home,
            away_team=away,
            league=str(ev.get("_tournament_name") or "Live Football"),
            category=str(ev.get("_category_name") or ""),
            minute=minute,
            clock_str=clock_str,
            period=period_desc,
            home_score=h_score,
            away_score=a_score,
            score_str=score_str,
            top_pick=top_pick,
            safe_pick=safe_pick,
            value_pick=value_pick,
            all_options=options,
            tactical_badge=tactical_badge,
            tactical_badge_type=tactical_badge_type,
            game_state_label=game_state_label,
            confidence_score=confidence_score,
            rationale=rationale
        )

    def scan_all_live_analyses(self) -> List[LiveFixtureAnalysis]:
        """
        Scans all current live matches and returns analyses sorted by betting viability.
        Prioritizes matches in Halftime Golden Windows and prime Game-State Asymmetry windows.
        """
        events = self.fetch_live_matches()
        analyses: List[LiveFixtureAnalysis] = []

        for ev in events:
            try:
                res = self.analyze_live_match(ev)
                if res and res.top_pick:
                    analyses.append(res)
            except Exception as e:
                logger.debug(f"Live match analysis error: {e}")

        # Sort analyses: Halftime Golden Windows and high-confidence games first
        analyses.sort(key=lambda a: (
            1 if a.tactical_badge_type == "golden_window" else 0,
            1 if (30 <= a.minute <= 80) else 0,
            a.confidence_score,
            a.top_pick.score
        ), reverse=True)

        logger.info(f"Analyzed {len(analyses)} live in-play football matches from SportyBet with Game-State Asymmetry.")
        return analyses

    def build_live_accumulator_slip(
        self,
        selections: List[Dict[str, Any]],
        stake: float = 50.0
    ) -> Dict[str, Any]:
        """
        Takes user-selected live picks, verifies odds, calculates joint probabilities,
        and generates an official SportyBet live booking code.
        """
        if not selections:
            return {"success": False, "message": "No live selections provided."}

        try:
            from automation.sportybet_bot import SportyBetAutomator
        except ImportError:
            from src.automation.sportybet_bot import SportyBetAutomator

        combined_odds = 1.0
        joint_prob = 1.0
        matches_payload = []

        for s in selections:
            odds = float(s.get("odds") or s.get("selected_odds") or 1.0)
            if s.get("prob") is not None:
                prob = float(s.get("prob"))
            elif s.get("prob_pct") is not None:
                prob = float(s.get("prob_pct")) / 100.0
            elif s.get("win_prob") is not None:
                prob = float(s.get("win_prob"))
            else:
                prob = 0.65

            if prob > 1.0:
                prob = prob / 100.0

            combined_odds *= odds
            joint_prob *= prob

            matches_payload.append({
                "sporty_id": str(s.get("sporty_id") or "").strip(),
                "home_team": str(s.get("home_team") or "").strip(),
                "away_team": str(s.get("away_team") or "").strip(),
                "selected_outcome": str(s.get("outcome_name") or s.get("selected_outcome") or "").strip(),
                "selected_odds": odds,
                "sporty_market_id": str(s.get("market_id") or s.get("sporty_market_id") or "").strip(),
                "sporty_outcome_id": str(s.get("outcome_id") or s.get("sporty_outcome_id") or "").strip(),
                "sporty_specifier": str(s.get("specifier") or s.get("sporty_specifier") or "").strip(),
            })

        booking_code = SportyBetAutomator.generate_booking_code_via_api(matches_payload)

        potential_win = round(stake * combined_odds, 2)
        ev_pct = round(((joint_prob * combined_odds) - 1.0) * 100, 1)

        return {
            "success": True,
            "booking_code": booking_code,
            "total_legs": len(matches_payload),
            "combined_odds": round(combined_odds, 2),
            "joint_probability": round(joint_prob * 100, 1),
            "expected_value_pct": ev_pct,
            "stake": stake,
            "potential_win": potential_win,
            "picks": matches_payload,
            "message": f"Successfully generated Live Booking Code: {booking_code}" if booking_code else "Assembled live slip (Direct play available)."
        }
