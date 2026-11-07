"""
AI Analytical Agent Module
Connected to 9Router / OpenAI / Gemini to assist web scraping match identification,
read raw predictions from multi-site sources, and fine-tune data for elite betting presentation.
"""

import os
import re
import json
import logging
import urllib.request
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from pathlib import Path

# Load environment
load_dotenv(dotenv_path=Path("./.env"), override=True)

logger = logging.getLogger(__name__)


class AIAnalyticalAgent:
    """
    AI-Powered Football Analyst & Scraper Guide Agent.
    - Guides the scraping engine to locate matches with irregular team naming
    - Interprets raw scraped predictions (Forebet, PredictZ)
    - Fine-tunes and formats statistical data into high-accuracy actionable betting analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("AI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("AI_BASE_URL") or "http://127.0.0.1:20128/v1").rstrip("/")
        self.model_name = model or os.getenv("AI_MODEL") or "claude-sonnet-5"
        self._is_active = bool(self.api_key)

        if self._is_active:
            logger.info(f"AI Analytical Agent initialized via 9Router/LLM at {self.base_url} (Model: {self.model_name})")

    def is_configured(self) -> bool:
        """Check if an AI API key is configured"""
        return self._is_active

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> Optional[str]:
        """Make an OpenAI-compatible completion request with stream=False"""
        if not self._is_active:
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 800,
            "stream": False
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=40) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Error calling LLM at {url}: {e}")
            return None

    def guide_match_search(self, target_home: str, target_away: str, candidates: List[str]) -> Optional[str]:
        """
        AI-guided match resolution:
        Helps the scraping engine identify which match in a scraped list corresponds to
        the booking code fixture (e.g. 'Shakhtar D' -> 'Shakhtar Donetsk').
        """
        if not candidates or not self._is_active:
            return None

        prompt = f"""Target Fixture to locate: '{target_home}' vs '{target_away}'.
Here is the candidate list of scraped matches from the website:
{json.dumps(candidates[:60], indent=2)}

Which candidate from the list represents the target fixture (even if team names have abbreviations or slight spelling differences)?
If a matching match exists, return EXACTLY that candidate string and nothing else.
If no match exists, return 'NONE'.
"""
        response = self._call_llm(prompt, system_prompt="You are an expert sports data normalizer. Return only the matching candidate string or NONE.")
        if response:
            clean_res = response.strip().strip('"').strip("'")
            if clean_res in candidates and clean_res != "NONE":
                logger.info(f"AI guided match match: '{target_home} vs {target_away}' -> '{clean_res}'")
                return clean_res
        return None

    def analyze_fixture(
        self,
        match_data: Dict[str, Any],
        consensus_stats: Optional[Dict[str, Any]] = None,
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune and synthesize scraped predictions, tactical stats, and odds into an elite presentation dossier.
        """
        home = match_data.get("home_team", "Home Team")
        away = match_data.get("away_team", "Away Team")
        odds_h = float(match_data.get("odds_home", 2.0))
        odds_d = float(match_data.get("odds_draw", 3.2))
        odds_a = float(match_data.get("odds_away", 3.5))
        pick = match_data.get("selected_outcome", "Home")
        pick_odds = float(match_data.get("selected_odds", odds_h))

        c_stats = consensus_stats or {}
        if hasattr(c_stats, "to_dict"):
            c_stats = c_stats.to_dict()

        forebet = c_stats.get("forebet", {})
        if not forebet and c_stats.get("has_forebet"):
            fb_p = c_stats.get("forebet_prob", {})
            forebet = {
                "prob_home": round(fb_p.get("home", 0.33) * 100, 1),
                "prob_draw": round(fb_p.get("draw", 0.33) * 100, 1),
                "prob_away": round(fb_p.get("away", 0.34) * 100, 1),
                "predicted_score": c_stats.get("forebet_score", ""),
                "avg_goals": c_stats.get("forebet_avg_goals", 2.5)
            }

        predictz = c_stats.get("predictz", {})
        if not predictz and c_stats.get("has_predictz"):
            predictz = {
                "prediction": c_stats.get("predictz_outcome", ""),
                "predicted_score": c_stats.get("predictz_score", ""),
                "home_form": "-".join(c_stats.get("home_form", [])) if isinstance(c_stats.get("home_form"), list) else str(c_stats.get("home_form", "")),
                "away_form": "-".join(c_stats.get("away_form", [])) if isinstance(c_stats.get("away_form"), list) else str(c_stats.get("away_form", ""))
            }

        consensus_prob = c_stats.get("consensus_prob", {})
        mo_bet = c_stats.get("matchoutlook_best_bet", "")
        mo_desc = c_stats.get("matchoutlook_desc", "")

        scoregpt = {
            "pick": c_stats.get("scoregpt_pick", ""),
            "score": c_stats.get("scoregpt_score", ""),
            "consensus": c_stats.get("scoregpt_consensus", ""),
            "panel": c_stats.get("scoregpt_panel", "")
        }
        betensured = {
            "tip": c_stats.get("betensured_tip", ""),
            "score": c_stats.get("betensured_score", ""),
            "prob": c_stats.get("betensured_prob", {})
        }

        ml_h2h = (ml_prediction or {}).get("h2h", {})
        ml_value = float(ml_h2h.get("value", 0.0))
        ml_prob = float(ml_h2h.get("model_probability", 0.5))
        kelly = float(ml_h2h.get("kelly_stake_percent", 1.0))
        multi_markets = (ml_prediction or {}).get("multi_markets", {})

        if not self._is_active:
            return self._generate_heuristic_analysis(
                home=home, away=away, odds=(odds_h, odds_d, odds_a),
                pick=pick, pick_odds=pick_odds,
                forebet=forebet, predictz=predictz,
                ml_value=ml_value, ml_prob=ml_prob, kelly=kelly,
                multi_markets=multi_markets, mo_bet=mo_bet
            )

        prompt = f"""You are an elite quantitative sports bettor and tactical football analyst.
Analyze the following match using this multi-site scraped intelligence and Poisson market models:

### FIXTURE DETAILS:
- Match: {home} (Home) vs {away} (Away)
- SportyBet 1X2 Odds: Home: {odds_h:.2f} | Draw: {odds_d:.2f} | Away: {odds_a:.2f}
- Betslip User Selection: {pick} @ {pick_odds:.2f}

### 5-SOURCE SCRAPED PREDICTIONS & EXPERT CONSENSUS:
1. Forebet (Algorithmic Mathematical Model):
   - 1X2 Probabilities: H: {forebet.get('prob_home', 'N/A')}% | D: {forebet.get('prob_draw', 'N/A')}% | A: {forebet.get('prob_away', 'N/A')}%
   - Predicted Score: {forebet.get('predicted_score', 'N/A')} | Avg Goals: {forebet.get('avg_goals', 'N/A')}
2. PredictZ (Form & Scoreline Engine):
   - Predicted Outcome: {predictz.get('prediction', 'N/A')} | Score: {predictz.get('predicted_score', 'N/A')}
   - Form Guide: Home: {predictz.get('home_form', 'N/A')} | Away: {predictz.get('away_form', 'N/A')}
3. ScoreGPT (Multi-Model Frontier AI Consensus - Claude, GPT, Grok):
   - AI Pick: {scoregpt.get('pick', 'N/A')} (Score: {scoregpt.get('score', 'N/A')})
   - Panel Verdict: {scoregpt.get('panel', 'N/A')} ({scoregpt.get('consensus', 'N/A')})
4. Betensured (Expert Analytical Platform):
   - Best Tip: {betensured.get('tip', 'N/A')} | Score: {betensured.get('score', 'N/A')}
   - Probabilities: {json.dumps(betensured.get('prob', {}))}
5. MatchOutlook (Alternative Market Tips):
   - Recommended Market: {mo_bet if mo_bet else 'General Pick'} ({mo_desc})

### QUANTITATIVE MULTI-MARKET OPTIONS (POISSON PROBABILITY & FAIR ODDS):
- Double Chance: {json.dumps(multi_markets.get('double_chance', {}))}
- Draw No Bet (DNB): {json.dumps(multi_markets.get('draw_no_bet', {}))}
- Totals (Over / Under): {json.dumps(multi_markets.get('totals', {}))}
- Both Teams To Score (BTTS): {json.dumps(multi_markets.get('btts', {}))}
- Asian Handicap: {json.dumps(multi_markets.get('asian_handicap', {}))}
- Clean Sheet: {json.dumps(multi_markets.get('clean_sheet', {}))}
- Top Correct Scores: {json.dumps(multi_markets.get('top_correct_scores', []))}
- Calculated Vig-Free Edge: {ml_value:+.2%}

### INSTRUCTIONS:
Provide an original, highly specific analysis tailored STRICTLY to this match's true odds and teams. DO NOT repeat identical phrases across fixtures.
Return JSON strictly with these exact keys:
{{
  "tactical_overview": "Specific 2-sentence tactical breakdown of {home} vs {away} addressing offensive and defensive matchup styles",
  "consensus_agreement": "High / Moderate / Divergent with exact comparison of the 5 sources above",
  "edge_evaluation": "Precise mathematical comparison between SportyBet's price ({pick_odds:.2f}) and true fair value",
  "risk_factors": ["Specific risk 1 for this game", "Specific risk 2 for this game"],
  "ai_verdict": "Clear decision (Value Bet, Fair Play, or Avoid)",
  "confidence_score": 8, // 1 to 10
  "recommended_stake": "{kelly:.1f}% Bankroll",
  "best_alternative_options": [
    {{"option": "e.g. Over 1.5 Goals or 1X", "market_type": "Totals / DC / AH", "rationale": "Clear statistical advantage over the user pick"}}
  ]
}}
"""

        raw_llm = self._call_llm(
            prompt,
            system_prompt="You are a professional sports bettor. Output strictly valid JSON without markdown formatting."
        )

        if raw_llm:
            try:
                clean_json = raw_llm.strip()
                if clean_json.startswith("```json"):
                    clean_json = clean_json[7:]
                if clean_json.startswith("```"):
                    clean_json = clean_json[3:]
                if clean_json.endswith("```"):
                    clean_json = clean_json[:-3]
                parsed = json.loads(clean_json.strip())
                parsed["ai_agent_status"] = f"active ({self.model_name})"
                return parsed
            except Exception as e:
                logger.error(f"Failed to parse LLM JSON response: {e}. Raw: {raw_llm[:100]}")

        return self._generate_heuristic_analysis(
            home=home, away=away, odds=(odds_h, odds_d, odds_a),
            pick=pick, pick_odds=pick_odds,
            forebet=forebet, predictz=predictz,
            ml_value=ml_value, ml_prob=ml_prob, kelly=kelly,
            multi_markets=multi_markets, mo_bet=mo_bet
        )

    def _generate_heuristic_analysis(
        self, home: str, away: str, odds: tuple,
        pick: str, pick_odds: float,
        forebet: Dict, predictz: Dict,
        ml_value: float, ml_prob: float, kelly: float,
        multi_markets: Optional[Dict] = None, mo_bet: str = ""
    ) -> Dict[str, Any]:
        """High-grade quantitative heuristic analysis fallback"""
        h_score = forebet.get("predicted_score") or predictz.get("predicted_score") or "1-1"
        is_value = ml_value > 0.05
        conf = min(9, max(5, int(ml_prob * 10) + (2 if is_value else 0)))

        alt_options = []
        if mo_bet:
            alt_options.append({"option": mo_bet, "market_type": "Specialized Platform", "rationale": "Recommended by MatchOutlook consensus"})
        if multi_markets:
            dc = multi_markets.get("double_chance", {})
            if "1X (Home/Draw)" in dc and dc["1X (Home/Draw)"]["prob"] > 0.65:
                alt_options.append({"option": "1X (Home/Draw)", "market_type": "Double Chance", "rationale": f"High safety margin ({dc['1X (Home/Draw)']['prob']*100:.0f}% probability)"})
            tot = multi_markets.get("totals", {})
            if "Over 1.5" in tot and tot["Over 1.5"]["prob"] > 0.70:
                alt_options.append({"option": "Over 1.5 Goals", "market_type": "Totals", "rationale": f"Strong offensive expectation ({tot['Over 1.5']['prob']*100:.0f}% probability)"})
            ah = multi_markets.get("asian_handicap", {})
            if "Home +0.5" in ah and ah["Home +0.5"]["prob"] > 0.60:
                alt_options.append({"option": "Home +0.5", "market_type": "Asian Handicap", "rationale": f"Covers win or draw ({ah['Home +0.5']['prob']*100:.0f}% probability)"})
            elif "Away +0.5" in ah and ah["Away +0.5"]["prob"] > 0.60:
                alt_options.append({"option": "Away +0.5", "market_type": "Asian Handicap", "rationale": f"Covers away win or draw ({ah['Away +0.5']['prob']*100:.0f}% probability)"})

        # Dynamic tactical breakdown based on specific odds and teams
        if odds[0] < 1.70:
            tac = f"{home} enter with significant market dominance at odds {odds[0]:.2f}. High home offensive pressure and possession dominance expected against {away}."
        elif odds[2] < 1.70:
            tac = f"{away} are strong market favorites at odds {odds[2]:.2f}. Counter-attacking vulnerabilities likely for {home} as the visitors dictate game pace."
        else:
            tac = f"Evenly matched tactical clash between {home} (odds {odds[0]:.2f}) and {away} (odds {odds[2]:.2f}). Midfield contest with consensus projecting scoreline around {h_score}."

        return {
            "tactical_overview": tac,
            "consensus_agreement": "High" if forebet.get("predicted_score") and predictz.get("prediction") else "Moderate",
            "edge_evaluation": f"SportyBet selection: {pick} @ {pick_odds:.2f} (Implied: {100.0/pick_odds if pick_odds>0 else 50:.1f}%). Model true probability: {ml_prob*100:.1f}% (Edge: {ml_value:+.2%}).",
            "risk_factors": [
                f"Variance in finishing efficiency and low-margin chance conversion",
                f"Tactical game state shift if early goal is conceded"
            ],
            "ai_verdict": f"Value Bet (Back {pick} @ {pick_odds:.2f})" if is_value else f"Fair Play: Market price aligns with probability",
            "confidence_score": conf,
            "recommended_stake": f"{max(1.0, min(5.0, kelly)):.1f}% Bankroll" if is_value else "1.0% Flat",
            "best_alternative_options": alt_options,
            "ai_agent_status": "heuristic_verified"
        }


# Singleton accessor
_ai_agent_instance: Optional[AIAnalyticalAgent] = None

def get_ai_agent(api_key: Optional[str] = None) -> AIAnalyticalAgent:
    global _ai_agent_instance
    if _ai_agent_instance is None or api_key:
        _ai_agent_instance = AIAnalyticalAgent(api_key=api_key)
    return _ai_agent_instance
