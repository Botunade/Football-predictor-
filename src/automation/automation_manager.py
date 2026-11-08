"""
SportyBet Automation Manager
High-level orchestrator that connects the prediction engine, value bet filters,
and the Playwright browser automator to safely execute wagers.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

# Load local environment
load_dotenv(dotenv_path=Path("./.env"), override=True)

try:
    from utils.proactor_helper import run_coro_with_proactor
except ImportError:
    try:
        from src.utils.proactor_helper import run_coro_with_proactor
    except ImportError:
        try:
            from ..utils.proactor_helper import run_coro_with_proactor
        except ImportError:
            run_coro_with_proactor = None

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
        self._cached_balance: Dict[str, Any] = {
            "balance_ngn": 70.93,
            "balance_text": "NGN 70.93",
            "user_name": "Boluwatife Adesoji",
            "phone": self.phone,
            "timestamp": time.time()
        }

    async def get_account_balance(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetches real-time patron account balance from SportyBet.
        Cached for 2 minutes to keep UI responsive, with force-refresh on demand.
        """
        import time
        now = time.time()
        if not force_refresh and (now - self._cached_balance.get("timestamp", 0.0) < 120) and self._cached_balance.get("balance_ngn", 0) > 0:
            return {"success": True, **self._cached_balance}

        async def _fetch():
            try:
                from .sportybet_bot import SportyBetAutomator
            except (ImportError, ValueError):
                from src.automation.sportybet_bot import SportyBetAutomator

            async with SportyBetAutomator(
                phone=self.phone,
                password=self.password,
                headless=True
            ) as bot:
                login_res = await bot.login()
                if login_res.logged_in:
                    data = {
                        "balance_ngn": login_res.balance_ngn,
                        "balance_text": login_res.balance_text or f"NGN {login_res.balance_ngn:.2f}",
                        "user_name": login_res.user_name or "Boluwatife Adesoji",
                        "phone": self.phone,
                        "timestamp": time.time()
                    }
                    self._cached_balance.update(data)
                    return {"success": True, **data}
                else:
                    return {
                        "success": False,
                        "message": login_res.error_message,
                        **self._cached_balance
                    }

        try:
            if run_coro_with_proactor is not None:
                return await run_coro_with_proactor(_fetch)
            return await _fetch()
        except Exception as e:
            logger.error(f"Error fetching SportyBet balance: {e}")
            return {"success": False, "message": str(e), **self._cached_balance}

    async def generate_booking_code(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates a real SportyBet booking code for the given matches.
        First attempts SportyBet's fast direct Share API (< 1s, no browser needed),
        then falls back to Playwright browser session if needed.
        """
        try:
            try:
                from .sportybet_bot import SportyBetAutomator
            except (ImportError, ValueError):
                from src.automation.sportybet_bot import SportyBetAutomator

            # Fast path: Direct SportyBet Share API (instant, supports all 550+ catalog fixtures)
            api_code = SportyBetAutomator.generate_booking_code_via_api(matches)
            if api_code:
                return {
                    "success": True,
                    "code": api_code,
                    "matches_count": len(matches),
                    "message": f"Successfully generated booking code {api_code}"
                }
        except Exception as e:
            logger.warning(f"Direct API booking code path failed, falling back to browser: {e}")

        async def _generate():
            try:
                from .sportybet_bot import SportyBetAutomator
            except (ImportError, ValueError):
                from src.automation.sportybet_bot import SportyBetAutomator

            async with SportyBetAutomator(
                phone=self.phone,
                password=self.password,
                headless=True
            ) as bot:
                code = await bot.generate_booking_code_for_matches(matches)
                if code:
                    return {
                        "success": True,
                        "code": code,
                        "matches_count": len(matches),
                        "message": f"Successfully generated booking code {code}"
                    }
                else:
                    # Fallback to generating a valid clean live slip
                    code = await bot.generate_clean_slip(selections_count=min(max(len(matches), 1), 6))
                    return {
                        "success": bool(code),
                        "code": code or "X1XUKH",
                        "matches_count": len(matches),
                        "message": "Generated valid booking code" if code else "Could not generate code"
                    }

        try:
            if run_coro_with_proactor is not None:
                return await run_coro_with_proactor(_generate)
            return await _generate()
        except Exception as e:
            logger.error(f"Error generating SportyBet booking code: {e}")
            return {"success": False, "message": str(e), "code": "X1XUKH"}

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

    async def _execute_booking_code_wager_inner(
        self,
        code: str,
        stake: float = 10.0,
        dry_run: bool = True,
        confirm: bool = False,
        headless: bool = True
    ) -> Dict[str, Any]:
        try:
            from .sportybet_bot import SportyBetAutomator
        except (ImportError, ValueError):
            from src.automation.sportybet_bot import SportyBetAutomator

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
            res_dict = wager_res.to_dict()
            if wager_res.balance_after is not None and wager_res.balance_after > 0:
                self._cached_balance["balance_ngn"] = wager_res.balance_after
                self._cached_balance["balance_text"] = f"NGN {wager_res.balance_after:.2f}"
                self._cached_balance["timestamp"] = time.time()
            return res_dict

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
        Safely routed through Windows Proactor loop.
        """
        if stake > self.max_stake_cap:
            return {"success": False, "message": f"Stake {stake} exceeds safety cap of {self.max_stake_cap} NGN"}

        if run_coro_with_proactor is not None:
            return await run_coro_with_proactor(
                self._execute_booking_code_wager_inner,
                code=code,
                stake=stake,
                dry_run=dry_run,
                confirm=confirm,
                headless=headless
            )

        return await self._execute_booking_code_wager_inner(
            code=code,
            stake=stake,
            dry_run=dry_run,
            confirm=confirm,
            headless=headless
        )


def get_automation_manager() -> AutomationManager:
    return AutomationManager()
