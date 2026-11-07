"""
Utility functions for data extraction and parsing
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MatchData:
    """Structured match data container"""
    home_team: str
    away_team: str
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    start_time: Optional[str] = None
    sport: str = "football"
    raw_data: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "odds_home": self.odds_home,
            "odds_draw": self.odds_draw,
            "odds_away": self.odds_away,
            "start_time": self.start_time or datetime.now(timezone.utc).isoformat(),
            "sport": self.sport
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MatchData":
        return cls(
            home_team=data.get("home_team", "Unknown"),
            away_team=data.get("away_team", "Unknown"),
            odds_home=float(data.get("odds_home", 0.0)),
            odds_draw=float(data.get("odds_draw", 0.0)),
            odds_away=float(data.get("odds_away", 0.0)),
            start_time=data.get("start_time"),
            sport=data.get("sport", "football"),
            raw_data=data
        )


def clean_team_name(name: str) -> str:
    """Clean and normalize team names"""
    if not name:
        return "Unknown"
    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name.strip())
    # Remove common prefixes/suffixes
    name = re.sub(r'^\[|\]$', '', name)
    return name


def extract_numeric(value: str, default: float = 0.0) -> float:
    """Extract numeric value from string, return default on failure"""
    if not value:
        return default
    try:
        # Handle various number formats
        value = str(value).strip()
        # Remove any non-numeric characters except . and -
        value = re.sub(r'[^\d.\-]', '', value)
        return float(value) if value else default
    except (ValueError, TypeError):
        return default


def parse_odds_from_text(elements_text: str) -> tuple:
    """
    Parse odds from text content
    Returns (home_odds, draw_odds, away_odds)
    """
    numbers = [float(n) for n in re.findall(r'[\d.]+', elements_text)]

    if len(numbers) == 3:
        return tuple(numbers)
    elif len(numbers) == 2:
        return (numbers[0], 0.0, numbers[1])
    else:
        return (0.0, 0.0, 0.0)


async def extract_odds_async(item) -> tuple:
    """
    Async extraction of odds from a match item (Playwright element)
    """
    odds_selectors = [
        ".odds-home, .home-odds, [data-testid='home-odds']",
        ".odds-draw, .draw-odds, [data-testid='draw-odds']",
        ".odds-away, .away-odds, [data-testid='away-odds']"
    ]

    odds = [0.0, 0.0, 0.0]

    for idx, selector in enumerate(odds_selectors):
        try:
            el = await item.query_selector(selector)
            if el:
                text = await el.text_content() if hasattr(el, 'text_content') else str(el)
                odds[idx] = extract_numeric(str(text).strip())
        except Exception:
            pass

    # Fallback: try ordered odds numbers
    if all(o == 0.0 for o in odds):
        try:
            odds_list = await item.query_selector_all(".m-odds-num, .odds, .odds-number, [data-testid='odds']")
            texts = []
            for el in odds_list:
                text = await el.text_content() if hasattr(el, 'text_content') else str(el)
                texts.append(extract_numeric(str(text).strip()))

            if len(texts) >= 3:
                odds = texts[:3]
            elif len(texts) == 2:
                odds = [texts[0], 0.0, texts[1]]
        except Exception:
            pass

    return tuple(odds)


def validate_match_data(match: Dict[str, Any]) -> bool:
    """Validate that match data has minimum required fields"""
    required_fields = ["home_team", "away_team"]

    if not isinstance(match, dict):
        return False

    for field in required_fields:
        if field not in match or not match[field]:
            return False

    # Validate team names are not placeholder values
    if "Unknown" in str(match.get("home_team", "")):
        return False
    if "Unknown" in str(match.get("away_team", "")):
        return False

    return True


def parse_match_data(raw_data: Dict[str, Any], sport: str = "football") -> Optional[Dict[str, Any]]:
    """
    Parse and validate raw scraped match data
    Returns structured match dict or None if invalid
    """
    if not raw_data:
        return None

    # Extract team names
    home_team = clean_team_name(raw_data.get("home_team", ""))
    away_team = clean_team_name(raw_data.get("away_team", ""))

    # Extract odds
    odds_home = extract_numeric(raw_data.get("odds_home"), 0.0)
    odds_draw = extract_numeric(raw_data.get("odds_draw"), 0.0)
    odds_away = extract_numeric(raw_data.get("odds_away"), 0.0)

    # Parse start time
    start_time = raw_data.get("start_time", datetime.now(timezone.utc).isoformat())

    parsed = {
        "home_team": home_team,
        "away_team": away_team,
        "odds_home": odds_home,
        "odds_draw": odds_draw,
        "odds_away": odds_away,
        "start_time": start_time,
        "sport": sport,
        "status": "upcoming"
    }

    if validate_match_data(parsed):
        return parsed

    return None


def log_scraping_result(matches: List[Dict], success: bool, message: str = ""):
    """Log scraping results with consistent format"""
    status = "✅" if success else "❌"
    count = len(matches) if matches else 0
    logger.info(f"{status} Scraped {count} matches. {message}")


def calculate_implied_probability(odds: float, remove_vig_margin: float = 0.0) -> float:
    """
    Calculate implied probability from decimal odds.
    Optionally normalizes by subtracting estimated vig margin.
    """
    if odds and odds > 1.0:
        raw_p = 1.0 / odds
        if remove_vig_margin > 0:
            raw_p = max(0.01, min(0.99, raw_p / (1.0 + remove_vig_margin)))
        return raw_p
    return 0.0


def calculate_fair_probabilities(odds_list: List[float]) -> List[float]:
    """
    Calculate fair true probabilities by removing bookmaker overround (margin/vig).
    Uses proportional normalization: P_fair_i = (1/odds_i) / sum(1/odds_k).
    """
    raw_probs = [1.0 / o if (o and o > 1.0) else 0.0 for o in odds_list]
    overround = sum(raw_probs)
    if overround <= 0:
        return [0.0] * len(odds_list)
    return [p / overround for p in raw_probs]


def calculate_kelly_stake(model_prob: float, odds: float, fraction: float = 0.25) -> float:
    """
    Calculate recommended stake percentage using Fractional Kelly Criterion.
    f* = (b*p - q) / b, where b = odds - 1, p = model_prob, q = 1 - p.
    Returns suggested bankroll percentage bounded [0.0, 10.0%].
    """
    if odds <= 1.0 or model_prob <= 0.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - model_prob
    f_star = (b * model_prob - q) / b
    if f_star <= 0:
        return 0.0
    return round(min(f_star * fraction * 100.0, 10.0), 2)


def calculate_value(model_prob: float, implied_prob: float) -> float:
    """Calculate value score: model probability - implied probability"""
    return model_prob - implied_prob


def detect_value_bet(value: float, threshold: float = 0.08) -> bool:
    """Determine if a bet has sufficient value"""
    return value > threshold