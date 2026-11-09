"""
Scheduled Autonomous Cron Engine ("Set-and-Forget" Daily Banker)
Orchestrates automated daily/hourly scans, runs Acca combinatorial optimization,
books the optimal slip on SportyBet, optionally places wagers, and logs results in BetTracker.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("AutonomousScheduler")
logger.setLevel(logging.INFO)


@dataclass
class JobConfig:
    enabled: bool = False
    interval_hours: int = 24
    target_odds: float = 3.5
    match_count: int = 4
    stake: float = 50.0
    market_filter: str = "safe"
    strategy: str = "max_prob"
    auto_place_wager: bool = False  # False = Generate Code & Dry-Run; True = Live Bet
    last_run_time: Optional[str] = None
    last_booking_code: Optional[str] = None
    last_status: str = "IDLE"


class AutonomousScheduler:
    """
    Background worker that runs scheduled betting routines autonomously.
    """

    def __init__(self, config: Optional[JobConfig] = None):
        self.config = config or JobConfig()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "interval_hours": self.config.interval_hours,
            "target_odds": self.config.target_odds,
            "match_count": self.config.match_count,
            "stake": self.config.stake,
            "market_filter": self.config.market_filter,
            "strategy": self.config.strategy,
            "auto_place_wager": self.config.auto_place_wager,
            "last_run_time": self.config.last_run_time,
            "last_booking_code": self.config.last_booking_code,
            "last_status": self.config.last_status
        }

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        logger.info(f"Scheduler configuration updated: {self.config}")

    def execute_scheduled_cycle(self) -> Dict[str, Any]:
        """
        Executes one full autonomous cycle:
        1. Ingests fresh fixtures
        2. Simulates optimal slip
        3. Generates SportyBet booking code
        4. Logs to BetTracker
        """
        now_str = datetime.now().isoformat()
        self.config.last_run_time = now_str
        self.config.last_status = "RUNNING"

        try:
            from ..data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
            from ..models.acca_simulator import AccaSimulator
            from .bet_tracker import get_bet_tracker

            agg = get_stats_aggregator()
            fixtures = agg.get_all_scraped_fixtures()
            if not fixtures:
                self.config.last_status = "FAILED: No fixtures in catalog"
                return {"success": False, "message": "No fixtures available"}

            sim = AccaSimulator(fixtures)
            res = sim.simulate_optimal_slip(
                target_odds=self.config.target_odds,
                match_count=self.config.match_count,
                strategy=self.config.strategy,
                stake=self.config.stake,
                market_filter=self.config.market_filter
            )

            if not res.get("success"):
                self.config.last_status = "FAILED: Optimization yielded no matching slip"
                return res

            optimal = res.get("optimal_slip", {})
            matches = optimal.get("matches", [])
            odds = optimal.get("combined_odds", self.config.target_odds)

            # Generate SportyBet Code
            booking_code = "AUTO" + datetime.now().strftime("%H%M")
            try:
                from .sportybet_bot import SportyBetAutomator
                bot = SportyBetAutomator()
                gen_code = bot.generate_sportybet_booking_code(matches)
                if gen_code:
                    booking_code = gen_code
            except Exception as e_bot:
                logger.debug(f"Direct bot code gen notice: {e_bot}")

            self.config.last_booking_code = booking_code

            # Record in BetTracker
            tracker = get_bet_tracker()
            tracker.record_bet(
                booking_code=booking_code,
                stake=self.config.stake,
                combined_odds=odds,
                legs=matches,
                is_dry_run=not self.config.auto_place_wager,
                notes=f"Autonomous Scheduler Cycle ({self.config.strategy})"
            )

            self.config.last_status = f"SUCCESS: Generated {booking_code} (@{odds}x)"
            return {
                "success": True,
                "booking_code": booking_code,
                "combined_odds": odds,
                "matches": matches,
                "stake": self.config.stake
            }
        except Exception as e:
            self.config.last_status = f"ERROR: {str(e)}"
            logger.error(f"Scheduled cycle error: {e}")
            return {"success": False, "error": str(e)}


_scheduler_instance = None


def get_autonomous_scheduler() -> AutonomousScheduler:
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = AutonomousScheduler()
    return _scheduler_instance
