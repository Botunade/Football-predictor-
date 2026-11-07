"""
Configuration module for Multi-Sport Predictor V4
Handles environment variables, default settings, and configuration management
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for ML models"""
    n_estimators: int = 100
    learning_rate: float = 0.05
    max_depth: int = 5
    random_state: int = 42

    # Feature selection
    feature_columns: List[str] = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
                "home_ppda", "away_ppda", "home_injury", "away_injury",
                "home_fatigue", "away_fatigue", "home_away_advantage",
                "rest_days_diff", "is_derby"
            ]


@dataclass
class ScraperConfig:
    """Configuration for web scraping"""
    headless: bool = True
    timeout: int = 60000  # milliseconds
    max_retries: int = 3
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    viewport_width: int = 1920
    viewport_height: int = 1080

    # SportyBet URLs
    base_url: str = "https://www.sportybet.com/ng/play/{}"
    fallback_urls: List[str] = None

    def __post_init__(self):
        if self.fallback_urls is None:
            self.fallback_urls = [
                "https://www.sportybet.com/ng/play/{}",
                "https://www.sportybet.com/en/play/{}",
            ]


@dataclass
class APIConfig:
    """Configuration for external statistical APIs (fixtures & statistics only; odds are scraped)"""
    api_key: Optional[str] = None
    timeout: int = 10
    max_concurrent_requests: int = 3

    # API-Sports endpoints (statistics and fixture data only)
    sports_api_base: str = "https://v3.football.api-sports.io"

    # Sport mappings
    sport_mappings: Dict[str, int] = None

    def __post_init__(self):
        if self.sport_mappings is None:
            self.sport_mappings = {
                "football": 39,  # EPL
                "basketball": 12,  # NBA
                "hockey": 57       # NHL
            }


@dataclass
class PredictionConfig:
    """Configuration for prediction outputs"""
    min_value_threshold: float = 0.08
    confidence_thresholds: Dict[str, float] = None
    output_format: str = "json"  # json, csv, web

    def __post_init__(self):
        if self.confidence_thresholds is None:
            self.confidence_thresholds = {
                "h2h": 0.55,
                "btts": 0.50,
                "over_25": 0.55
            }


def load_config() -> tuple[ModelConfig, ScraperConfig, APIConfig, PredictionConfig]:
    """Load configuration from environment variables"""

    # Load from environment
    api_key = os.getenv("API_KEY")

    # Create configs
    model_cfg = ModelConfig(
        n_estimators=int(os.getenv("MODEL_N_ESTIMATORS", 100)),
        learning_rate=float(os.getenv("MODEL_LR", 0.05)),
        max_depth=int(os.getenv("MODEL_MAX_DEPTH", 5)),
        random_state=int(os.getenv("MODEL_RANDOM_STATE", 42))
    )

    scraper_cfg = ScraperConfig(
        headless=os.getenv("SCRAPER_HEADLESS", "true").lower() == "true",
        timeout=int(os.getenv("SCRAPER_TIMEOUT", 60000)),
        max_retries=int(os.getenv("SCRAPER_MAX_RETRIES", 3))
    )

    api_cfg = APIConfig(
        api_key=api_key,
        timeout=int(os.getenv("API_TIMEOUT", 10)),
        max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", 3))
    )

    pred_cfg = PredictionConfig(
        min_value_threshold=float(os.getenv("MIN_VALUE_THRESHOLD", 0.08)),
        output_format=os.getenv("OUTPUT_FORMAT", "json")
    )

    return model_cfg, scraper_cfg, api_cfg, pred_cfg


# Environment file path
ENV_FILE = Path(__file__).parent.parent / ".env"