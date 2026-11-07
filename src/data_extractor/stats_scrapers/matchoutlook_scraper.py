"""
MatchOutlook Scraper Module
Extracts specialized match predictions, totals (Over/Under), and betting market tips
from MatchOutlook (https://www.matchoutlook.com)
"""

import asyncio
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

logger = logging.getLogger(__name__)


@dataclass
class MatchOutlookPrediction:
    """Structured prediction record from MatchOutlook"""
    home_team: str
    away_team: str
    league: str
    kickoff: str
    best_bet: str
    description: str
    scraped_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MatchOutlookScraper:
    """
    Playwright-based scraper for MatchOutlook daily predictions.
    Focuses on alternate betting options (Over 1.5, Over 2.5, BTTS, Double Chance).
    """

    TARGET_URL = "https://www.matchoutlook.com/todays-football-predictions/"

    def __init__(self, headless: bool = True, timeout: int = 35000):
        self.headless = headless
        self.timeout = timeout
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        self._context = await self._browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if hasattr(self, '_playwright') and self._playwright:
            await self._playwright.stop()

    async def scrape_predictions(self) -> List[MatchOutlookPrediction]:
        """Scrape predictions from MatchOutlook"""
        predictions: List[MatchOutlookPrediction] = []
        page = await self._context.new_page()

        try:
            logger.info(f"Navigating to MatchOutlook: {self.TARGET_URL}")
            await page.goto(self.TARGET_URL, wait_until="domcontentloaded", timeout=self.timeout)
            await page.wait_for_timeout(3000)

            body_text = await page.inner_text("body")
            lines = [l.strip() for l in body_text.split("\n") if l.strip()]

            # Parse lines looking for fixture patterns
            i = 0
            current_league = "General"
            while i < len(lines):
                line = lines[i]

                # Check for league line
                if any(k in line.lower() for k in ["league", "cup", "pokalen", "copa", "division", "serie", "bundesliga", "primera"]):
                    current_league = line

                # Check for fixture line
                if " vs " in line:
                    parts = line.split(" vs ", 1)
                    if len(parts) == 2:
                        home = parts[0].strip()
                        away = parts[1].strip()

                        best_bet = "General Pick"
                        desc = ""
                        kickoff = ""

                        # Check preceding line for time
                        if i > 0 and re.match(r'^\d{1,2}:\d{2}$', lines[i-1]):
                            kickoff = lines[i-1]

                        # Check subsequent lines for Best Bet
                        j = i + 1
                        while j < min(len(lines), i + 4):
                            if "best bet" in lines[j].lower():
                                best_bet = lines[j].replace("Best Bet:", "").strip()
                                if j + 1 < len(lines):
                                    desc = lines[j+1]
                                break
                            j += 1

                        pred = MatchOutlookPrediction(
                            home_team=home,
                            away_team=away,
                            league=current_league,
                            kickoff=kickoff,
                            best_bet=best_bet,
                            description=desc,
                            scraped_at=datetime.now(timezone.utc).isoformat()
                        )
                        predictions.append(pred)
                i += 1

            logger.info(f"Successfully scraped {len(predictions)} MatchOutlook predictions")
        except Exception as e:
            logger.error(f"MatchOutlook scraping error: {e}")
        finally:
            await page.close()

        return predictions


# Standalone runner for testing
if __name__ == "__main__":
    async def main():
        async with MatchOutlookScraper(headless=True) as scraper:
            preds = await scraper.scrape_predictions()
            print(f"Total MatchOutlook scraped: {len(preds)}")
            for p in preds:
                print(f"  {p.home_team} vs {p.away_team} | Best Bet: {p.best_bet} ({p.description})")

    asyncio.run(main())
