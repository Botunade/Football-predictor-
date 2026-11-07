"""
PredictZ Scraper Module
Extracts match predictions, predicted scorelines, team form (W/D/L),
and league standings directly from PredictZ via Playwright.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger(__name__)


@dataclass
class PredictZPrediction:
    """Structured prediction extracted from PredictZ"""
    home_team: str
    away_team: str
    prediction: str  # e.g., "Home Win 2-1" or "Draw 1-1"
    predicted_score: str  # e.g., "2-1"
    predicted_outcome: str  # "Home", "Draw", "Away"
    home_form: List[str] = field(default_factory=list)  # e.g. ["W", "D", "L", "W", "W"]
    away_form: List[str] = field(default_factory=list)
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    league: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "prediction": self.prediction,
            "predicted_score": self.predicted_score,
            "predicted_outcome": self.predicted_outcome,
            "home_form": self.home_form,
            "away_form": self.away_form,
            "odds_home": self.odds_home,
            "odds_draw": self.odds_draw,
            "odds_away": self.odds_away,
            "league": self.league
        }


class PredictZScraper:
    """
    Playwright-based scraper for PredictZ predictions, form, and standings.
    """

    DEFAULT_LEAGUE_URLS = [
        ("Premier League", "https://www.predictz.com/predictions/england/premier-league/"),
        ("Champions League", "https://www.predictz.com/predictions/champions-league/"),
        ("Europa League", "https://www.predictz.com/predictions/europa-league/"),
        ("La Liga", "https://www.predictz.com/predictions/spain/la-liga/"),
        ("Bundesliga", "https://www.predictz.com/predictions/germany/bundesliga/"),
        ("Serie A", "https://www.predictz.com/predictions/italy/serie-a/"),
        ("Ligue 1", "https://www.predictz.com/predictions/france/ligue-1/")
    ]

    def __init__(self, headless: bool = True, timeout: int = 25000):
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

    async def scrape_league(self, league_name: str, url: str) -> List[PredictZPrediction]:
        """Scrape predictions for a single league"""
        predictions: List[PredictZPrediction] = []
        if not self._context:
            return predictions

        page = await self._context.new_page()
        try:
            logger.info(f"Scraping PredictZ [{league_name}]: {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
            await page.wait_for_timeout(2000)

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            tables = soup.find_all('table')
            if not tables:
                return predictions

            # First table is typically the predictions table
            pred_table = tables[0]
            rows = pred_table.find_all('tr')

            for r in rows:
                cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
                row_text = " , ".join(cells)

                # Identify match row by presence of ' vs '
                if " vs " in row_text:
                    parsed = self._parse_table_row(cells, league_name)
                    if parsed:
                        predictions.append(parsed)

        except Exception as e:
            logger.error(f"Failed scraping PredictZ league {league_name}: {e}")
        finally:
            await page.close()

        return predictions

    async def scrape_all_leagues(self) -> List[PredictZPrediction]:
        """Scrape predictions across major European competitions"""
        all_preds: List[PredictZPrediction] = []
        for name, url in self.DEFAULT_LEAGUE_URLS:
            preds = await self.scrape_league(name, url)
            all_preds.extend(preds)
        logger.info(f"Total PredictZ predictions scraped: {len(all_preds)}")
        return all_preds

    def _parse_table_row(self, cells: List[str], league: str) -> Optional[PredictZPrediction]:
        """Parse cells from a PredictZ prediction row"""
        try:
            # Find the match cell (contains ' vs ')
            match_idx = -1
            for i, c in enumerate(cells):
                if " vs " in c:
                    match_idx = i
                    break

            if match_idx == -1:
                return None

            match_str = cells[match_idx]
            parts = match_str.split(" vs ")
            if len(parts) != 2:
                return None
            home_team = parts[0].strip()
            away_team = parts[1].strip()

            # Prediction is typically in cells before the match cell
            pred_str = cells[0] if match_idx > 0 else ""
            
            # Extract score
            score_match = re.search(r'(\d+)\s*-\s*(\d+)', pred_str)
            predicted_score = score_match.group(0) if score_match else "1-1"

            # Determine outcome
            pred_lower = pred_str.lower()
            if "home" in pred_lower or "win" in pred_lower and not ("away" in pred_lower or "draw" in pred_lower):
                predicted_outcome = "Home"
            elif "away" in pred_lower:
                predicted_outcome = "Away"
            elif "draw" in pred_lower:
                predicted_outcome = "Draw"
            else:
                if score_match:
                    h, a = int(score_match.group(1)), int(score_match.group(2))
                    predicted_outcome = "Home" if h > a else ("Away" if a > h else "Draw")
                else:
                    predicted_outcome = "Home"

            # Extract form (letters W, D, L)
            home_form = [c for c in cells[1:match_idx] if c in ["W", "D", "L"]]
            away_form = [c for c in cells[match_idx+1:match_idx+6] if c in ["W", "D", "L"]]

            # Extract odds from the end of row
            odds = [float(c) for c in cells if re.match(r'^\d+\.\d{2}$', c)]
            odds_h = odds[0] if len(odds) >= 3 else 0.0
            odds_d = odds[1] if len(odds) >= 3 else 0.0
            odds_a = odds[2] if len(odds) >= 3 else 0.0

            return PredictZPrediction(
                home_team=home_team,
                away_team=away_team,
                prediction=pred_str,
                predicted_score=predicted_score,
                predicted_outcome=predicted_outcome,
                home_form=home_form,
                away_form=away_form,
                odds_home=odds_h,
                odds_draw=odds_d,
                odds_away=odds_a,
                league=league
            )
        except Exception as e:
            logger.debug(f"Failed parsing PredictZ cells {cells}: {e}")
            return None
