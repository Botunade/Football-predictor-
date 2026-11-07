"""
Forebet Scraper Module
Extracts mathematical predictions, 1X2 probabilities, predicted exact scorelines,
and expected goal totals directly from Forebet via Playwright.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

logger = logging.getLogger(__name__)


@dataclass
class ForebetPrediction:
    """Structured prediction extracted from Forebet"""
    home_team: str
    away_team: str
    prob_home: float  # e.g., 0.45
    prob_draw: float  # e.g., 0.28
    prob_away: float  # e.g., 0.27
    predicted_score: str  # e.g., "2-1"
    avg_goals: float  # e.g., 2.75
    tip: str  # "1", "X", "2", "1X", etc.
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    league: str = ""
    match_time: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prob_home": round(self.prob_home, 4),
            "prob_draw": round(self.prob_draw, 4),
            "prob_away": round(self.prob_away, 4),
            "predicted_score": self.predicted_score,
            "avg_goals": round(self.avg_goals, 2),
            "tip": self.tip,
            "odds_home": self.odds_home,
            "odds_draw": self.odds_draw,
            "odds_away": self.odds_away,
            "league": self.league,
            "match_time": self.match_time
        }


class ForebetScraper:
    """
    Playwright-based scraper for Forebet algorithmic predictions.
    """

    BASE_URLS = [
        "https://www.forebet.com/en/football-predictions",
        "https://www.forebet.com/en/football-predictions/predictions-1x2",
        "https://www.forebet.com/en/football-predictions/under-over-25-goals"
    ]

    def __init__(self, headless: bool = True, timeout: int = 45000):
        self.headless = headless
        self.timeout = timeout
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-infobars'
            ]
        )
        self._context = await self._browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if hasattr(self, '_playwright') and self._playwright:
            await self._playwright.stop()

    async def scrape_predictions(self, max_matches: int = 200) -> List[ForebetPrediction]:
        """Scrape current predictions from Forebet"""
        predictions: List[ForebetPrediction] = []
        if not self._context:
            logger.error("ForebetScraper must be used as an async context manager.")
            return predictions

        page = await self._context.new_page()
        try:
            url = self.BASE_URLS[0]
            logger.info(f"Navigating to Forebet: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
            await page.wait_for_timeout(2500)

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            rows = soup.select(".rcnt, .tr_0, .tr_1")
            logger.info(f"Forebet returned {len(rows)} raw match rows")

            for row in rows[:max_matches]:
                try:
                    pred = self._parse_row(row)
                    if pred:
                        predictions.append(pred)
                except Exception as e:
                    logger.debug(f"Error parsing Forebet row: {e}")
                    continue

        except Exception as e:
            logger.error(f"Forebet scraping failed: {e}")
        finally:
            await page.close()

        logger.info(f"Successfully scraped {len(predictions)} Forebet predictions")
        return predictions

    def _parse_row(self, row) -> Optional[ForebetPrediction]:
        """Parse a single row from Forebet"""
        home_el = row.select_one(".homeTeam")
        away_el = row.select_one(".awayTeam")
        if not home_el or not away_el:
            return None

        home_team = home_el.get_text(strip=True)
        away_team = away_el.get_text(strip=True)
        if not home_team or not away_team:
            return None

        # Extract probabilities (1, X, 2)
        prob_spans = row.select(".fprc span")
        p_home, p_draw, p_away = 0.33, 0.33, 0.34
        if len(prob_spans) >= 3:
            try:
                p_home = float(re.sub(r'[^\d.]', '', prob_spans[0].get_text())) / 100.0
                p_draw = float(re.sub(r'[^\d.]', '', prob_spans[1].get_text())) / 100.0
                p_away = float(re.sub(r'[^\d.]', '', prob_spans[2].get_text())) / 100.0
            except Exception:
                pass

        # Predicted exact score
        score_el = row.select_one(".ex_sc")
        predicted_score = score_el.get_text(strip=True) if score_el else "1-1"

        # Average expected goals
        avg_el = row.select_one(".avg_sc")
        avg_goals = 2.5
        if avg_el:
            try:
                avg_goals = float(re.sub(r'[^\d.]', '', avg_el.get_text()))
            except Exception:
                pass

        # Prediction tip
        tip_el = row.select_one(".forepred")
        tip = tip_el.get_text(strip=True) if tip_el else ("1" if p_home > p_away else "2")

        # League and time
        league_el = row.select_one(".shortTag, .schema_tag")
        league = league_el.get_text(strip=True) if league_el else ""

        time_el = row.select_one(".date_stat")
        match_time = time_el.get_text(strip=True) if time_el else ""

        return ForebetPrediction(
            home_team=home_team,
            away_team=away_team,
            prob_home=p_home,
            prob_draw=p_draw,
            prob_away=p_away,
            predicted_score=predicted_score,
            avg_goals=avg_goals,
            tip=tip,
            league=league,
            match_time=match_time
        )


async def get_forebet_predictions() -> List[ForebetPrediction]:
    """Convenience helper to extract Forebet predictions"""
    async with ForebetScraper(headless=True) as scraper:
        return await scraper.scrape_predictions()
