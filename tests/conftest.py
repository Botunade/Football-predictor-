"""
Pytest configuration and fixtures for Multi-Sport Predictor V4
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_match_data():
    """Sample match data for testing"""
    return {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "odds_home": 2.1,
        "odds_draw": 3.5,
        "odds_away": 4.2,
        "start_time": "2024-01-15T15:00:00Z",
        "sport": "football"
    }


@pytest.fixture
def sample_features():
    """Sample ML features for testing"""
    return {
        "home_xG": 2.0,
        "away_xG": 1.5,
        "home_xGA": 1.0,
        "away_xGA": 1.2,
        "home_xGD": 1.0,
        "away_xGD": 0.3,
        "home_ppda": 10.0,
        "away_ppda": 12.0,
        "home_injury": 0.1,
        "away_injury": 0.2,
        "home_fatigue": 0.15,
        "away_fatigue": 0.25,
        "home_away_advantage": 1.1,
        "rest_days_diff": 0,
        "is_derby": 0,
        "odds_home": 2.1,
        "odds_draw": 3.5,
        "odds_away": 4.2
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical match data for model training"""
    return [
        {
            "home_team": "Team A",
            "away_team": "Team B",
            "home_xG": 2.0,
            "away_xG": 1.5,
            "home_xGA": 1.0,
            "away_xGA": 1.2,
            "home_xGD": 1.0,
            "away_xGD": 0.3,
            "home_ppda": 10.0,
            "away_ppda": 12.0,
            "home_injury": 0.1,
            "away_injury": 0.2,
            "home_fatigue": 0.15,
            "away_fatigue": 0.25,
            "home_away_advantage": 1.1,
            "rest_days_diff": 0,
            "is_derby": 0,
            "actual_result": "home"
        },
        {
            "home_team": "Team C",
            "away_team": "Team D",
            "home_xG": 1.2,
            "away_xG": 1.8,
            "home_xGA": 1.5,
            "away_xGA": 0.9,
            "home_xGD": -0.3,
            "away_xGD": 0.9,
            "home_ppda": 15.0,
            "away_ppda": 8.0,
            "home_injury": 0.3,
            "away_injury": 0.1,
            "home_fatigue": 0.25,
            "away_fatigue": 0.15,
            "home_away_advantage": 1.1,
            "rest_days_diff": 1,
            "is_derby": 0,
            "actual_result": "away"
        },
        {
            "home_team": "Team E",
            "away_team": "Team F",
            "home_xG": 1.5,
            "away_xG": 1.5,
            "home_xGA": 1.2,
            "away_xGA": 1.2,
            "home_xGD": 0.3,
            "away_xGD": 0.3,
            "home_ppda": 12.0,
            "away_ppda": 12.0,
            "home_injury": 0.15,
            "away_injury": 0.15,
            "home_fatigue": 0.2,
            "away_fatigue": 0.2,
            "home_away_advantage": 1.1,
            "rest_days_diff": 0,
            "is_derby": 0,
            "actual_result": "draw"
        }
    ]


@pytest.fixture
def mock_browser():
    """Mock browser for testing scraper without Playwright"""
    pass


# Configure pytest-asyncio if needed
def pytest_configure(config):
    """Configure pytest with async support"""
    try:
        import pytest_asyncio
    except ImportError:
        pass  # pytest-asyncio is optional