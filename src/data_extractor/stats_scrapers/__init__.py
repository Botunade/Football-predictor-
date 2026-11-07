"""
Statistical scrapers package: Forebet, PredictZ, and Unified Stats Aggregator
"""

from .forebet_scraper import ForebetScraper, ForebetPrediction, get_forebet_predictions
from .predictz_scraper import PredictZScraper, PredictZPrediction
from .matchoutlook_scraper import MatchOutlookScraper, MatchOutlookPrediction
from .stats_aggregator import StatsAggregator, AggregatedStats, get_stats_aggregator

__all__ = [
    "ForebetScraper",
    "ForebetPrediction",
    "get_forebet_predictions",
    "PredictZScraper",
    "PredictZPrediction",
    "MatchOutlookScraper",
    "MatchOutlookPrediction",
    "StatsAggregator",
    "AggregatedStats",
    "get_stats_aggregator"
]
