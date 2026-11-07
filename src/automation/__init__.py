"""
Automation Package for Football Predictor
Provides SportyBet betting automation and slip management.
"""

from .sportybet_bot import (
    SportyBetAutomator,
    LoginResult,
    BetslipState,
    BetslipItem,
    WagerResult
)

__all__ = [
    "SportyBetAutomator",
    "LoginResult",
    "BetslipState",
    "BetslipItem",
    "WagerResult"
]
