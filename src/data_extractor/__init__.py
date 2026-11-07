"""
Data extraction module for sports prediction
Provides web scraping and API integration capabilities
"""

from .sportybet_scraper import extract_sporty_code, SportyBetScraper

# Import from multi_site_scraper (enhanced version)
from .multi_site_scraper import (
    # Core classes
    ScraperConfig,
    ScraperSiteConfig,
    Bookmaker,
    ScrapeResult,
    MultiSiteScraper,
    # Core functions
    extract_all_sites,
    save_results_to_json,
    extract_from_url_sync,
    # Site-specific convenience functions
    extract_sportybet_ng,
    extract_bet9ja,
    extract_nairabet,
    # Constants and utilities
    SITE_CONFIGS,
    ScrapingError,
    log_scraping_result as scraper_log_scraping_result,
    # Legacy exports (renamed for backward compatibility)
    legacy_extract_from_all_sites as extract_from_all_sites_legacy,
    legacy_extract_sporty_code,
)

from .scraper_utils import (
    parse_match_data,
    extract_odds_async,
    validate_match_data,
    calculate_implied_probability,
    calculate_value,
    detect_value_bet,
    MatchData,
)

from .code_reader import (
    extract_betting_code,
    extract_betting_code_async,
    read_sporty_code,
    CodeReader,
    ErrorCode,
    validate_code_format,
    Match,
    Odds,
    Metadata,
    ExtractionDetails,
)

# Use the log_scraping_result from multi_site_scraper (preferred version)
log_scraping_result = scraper_log_scraping_result

# Backward compatibility alias
primary_extract = legacy_extract_sporty_code

__all__ = [
    # Legacy exports from sportybet_scraper
    "extract_sporty_code",
    "SportyBetScraper",

    # Multi-site scraper (enhanced)
    "ScraperConfig",
    "ScraperSiteConfig",
    "Bookmaker",
    "ScrapeResult",
    "MultiSiteScraper",
    "extract_all_sites",
    "save_results_to_json",
    "extract_from_url_sync",
    "extract_sportybet_ng",
    "extract_bet9ja",
    "extract_nairabet",
    "SITE_CONFIGS",
    "ScrapingError",
    "log_scraping_result",
    "primary_extract",

    # Utilities from scraper_utils
    "parse_match_data",
    "extract_odds_async",
    "validate_match_data",
    "calculate_implied_probability",
    "calculate_value",
    "detect_value_bet",
    "MatchData",

    # Code reader module
    "extract_betting_code",
    "extract_betting_code_async",
    "read_sporty_code",
    "CodeReader",
    "ErrorCode",
    "validate_code_format",
    "Match",
    "Odds",
    "Metadata",
    "ExtractionDetails",
]