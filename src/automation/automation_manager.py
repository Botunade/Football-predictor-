"""
SportyBet Automation Manager
High-level orchestrator that connects the prediction engine, value bet filters,
and the Playwright browser automator to safely execute wagers.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load local environment
load_dotenv(dotenv_path=Path("./.env"), override=True)

logger = logging.getLogger(__name__)


class AutomationManager:
    """
    Manager for SportyBet Automated Wagering:
    - Enforces safety limits (e.g. maximum stake cap, dry-run safety gates)
    - Selects the top AI-verified value bets
    - Coordinates with SportyBetAutomator for browser execution
    """

    def __init__(self, max_stake_cap: float = 50.0):
        self.max_stake_cap = max_stake_cap
        self.phone = os.getenv("SPORTY_PHONE", "")
        self.password = os.getenv("SPORTY_PASSWORD", "")
        self.default_stake = float(os.getenv("SPORTY_STAKE_TEST", "10.0"))

    def filter_elite_wagers(self, analyzed_predictions: List[Dict[str, Any]], max_picks: int = 3) -> List[Dict[str, Any]]:
        """
        Selects the most statistically robust fixtures:
        - High AI Confidence (>= 7/10)
        - Positive vig-free edge (> +5%)
        - Non-avoid AI verdict
        """
        candidates = []
        for p in analyzed_predictions:
            match = p.get("match", {})
            pred = (p.get("prediction") or {}).get("h2h", {})
            ai = p.get("ai_analysis") or {}
            enrich = p.get("enrichment") or {}

            conf = ai.get("confidence_score", 5)
            verdict = str(ai.get("ai_verdict", "")).lower()
            edge = float(pred.get("value", 0.0))
            is_bogey = (enrich.get("h2h") or {}).get("is_bogey_team", False)

            # Strict risk filter: avoid traps and low confidence
            if "avoid" in verdict or is_bogey:
                continue

            score = (conf * 10) + (edge * 100)
            candidates.append({
                "match": match,
                "home_team": match.get("home_team"),
                "away_team": match.get("away_team"),
                "selected_outcome": match.get("selected_outcome", "Home"),
                "selected_odds": float(match.get("selected_odds") or match.get("odds_home") or 2.0),
                "edge": edge,
                "confidence": conf,
                "verdict": ai.get("ai_verdict"),
                "ranking_score": score
            })

        # Sort by composite ranking score
        candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
        return candidates[:max_picks]

    async def run_automated_bet(
        self,
        picks: List[Dict[str, Any]],
        stake: float = 10.0,
        dry_run: bool = False,
        headless: bool = False
    ) -> Dict[str, Any]:
        """
        Executes an automated wagering run on SportyBet Nigeria:
        - Validates stake against safety cap
        - Delegates browser actions to SportyBetAutomator
        """
        if stake > self.max_stake_cap:
            raise ValueError(f"Requested stake {stake} NGN exceeds safety cap of {self.max_stake_cap} NGN")

        if not self.phone or not self.password:
            return {
                "status": "error",
                "message": "SportyBet phone or password not configured in .env"
            }

        try:
            from .sportybet_bot import SportyBetAutomator
        except (ImportError, ValueError):
            try:
                from src.automation.sportybet_bot import SportyBetAutomator
            except ImportError:
                return {
                    "status": "error",
                    "message": "SportyBetAutomator module is being built by the automation engineer subagent."
                }

        automator = SportyBetAutomator(
            phone=self.phone,
            password=self.password,
            headless=headless,
            stake=stake,
            dry_run=dry_run
        )

        async with automator:
            login_res = await automator.login()
            if not login_res.logged_in:
                return {
                    "success": False,
                    "message": f"Login failed: {login_res.error_message}",
                    "details": login_res.to_dict()
                }
            return (await automator.execute_wager(stake=stake, dry_run=dry_run)).to_dict()

    async def execute_booking_code_wager(
        self,
        code: str,
        stake: float = 10.0,
        dry_run: bool = True,
        confirm: bool = False,
        headless: bool = True
    ) -> Dict[str, Any]:
        """
        Loads an active booking code into SportyBet, sets 10 NGN stake,
        and executes either dry-run validation or live wager placement.
        """
        if stake > self.max_stake_cap:
            return {"success": False, "message": f"Stake {stake} exceeds safety cap of {self.max_stake_cap} NGN"}

        from .sportybet_bot import SportyBetAutomator

        async with SportyBetAutomator(
            phone=self.phone,
            password=self.password,
            headless=headless,
            stake=stake,
            dry_run=dry_run
        ) as bot:
            login_res = await bot.login()
            if not login_res.logged_in:
                return {
                    "success": False,
                    "mode": "dry_run" if dry_run else "live",
                    "message": f"Login failed: {login_res.error_message or 'Unknown error'}",
                    "details": login_res.to_dict()
                }

            slip_state = await bot.load_booking_code(code)
            if slip_state.error or slip_state.has_unavailable:
                return {
                    "success": False,
                    "mode": "dry_run" if dry_run else "live",
                    "message": slip_state.error or "Betslip contains unavailable or expired matches",
                    "details": slip_state.to_dict()
                }

            wager_res = await bot.execute_wager(stake=stake, dry_run=dry_run, confirm=confirm)
            return wager_res.to_dict()


def get_automation_manager() -> AutomationManager:
    return AutomationManager()
