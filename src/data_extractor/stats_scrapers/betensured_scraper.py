"""
Betensured Scraper Module
Extracts high-accuracy algorithmic predictions, consensus probabilities, 
market tips (1X, X2, Over/Under, BTTS), and predicted scorelines from Betensured (https://www.betensured.com)
"""

import asyncio
import logging
import re
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class BetensuredPrediction:
    """Structured prediction record from Betensured"""
    home_team: str
    away_team: str
    league: str
    prob_home: int = 0
    prob_draw: int = 0
    prob_away: int = 0
    tip: str = ""
    predicted_score: str = ""
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    scraped_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BetensuredScraper:
    """
    Fast HTTP and BeautifulSoup scraper for Betensured daily predictions.
    Extracts 1X2 probabilities, predicted scores, and alternative market tips.
    """

    TARGET_URL = "https://www.betensured.com/"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def scrape_predictions(self) -> List[BetensuredPrediction]:
        """Fetch and parse all football predictions from Betensured"""
        return await asyncio.to_thread(self._scrape_sync)

    def _scrape_sync(self) -> List[BetensuredPrediction]:
        predictions: List[BetensuredPrediction] = []
        try:
            req = urllib.request.Request(
                self.TARGET_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                }
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                html = response.read().decode("utf-8", errors="ignore")

            soup = BeautifulSoup(html, "html.parser")
            tables = soup.find_all("table")

            now_str = datetime.now(timezone.utc).isoformat()

            for table in tables:
                rows = table.find_all("tr")
                for row in rows:
                    tds = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                    if not tds:
                        continue

                    row_text = " | ".join(tds)
                    # Check for fixture line
                    # Format A: ['UCLFeyenoord vs Inter', 'OVER 2.5', '', 'OVER 1.5', 'Statistics', '']
                    # Format B: ['00:15', 'Argentina: Liga Profesional...Atl. TucumanVSArgentinos Jrs', '126%X32%242%', '3.02', '3.02', '2.57', 'X2', '1 - 2', '']
                    
                    # Pattern 1: Featured matches
                    if len(tds) >= 4 and (" vs " in tds[0].lower() or " vs. " in tds[0].lower()):
                        fixture_raw = tds[0]
                        # Remove league prefix like UCL, ECH, SWE (2 to 4 capital letters at start followed by capital letter)
                        cleaned_fixture = re.sub(r'^[A-Z]{2,4}(?=[A-Z][a-z])', '', fixture_raw).strip()
                        parts = re.split(r'\s+vs\.?\s+', cleaned_fixture, flags=re.IGNORECASE)
                        if len(parts) == 2:
                            home, away = parts[0].strip(), parts[1].strip()
                            tip = tds[1].strip() or tds[3].strip()
                            pred = BetensuredPrediction(
                                home_team=home,
                                away_team=away,
                                league="Featured",
                                tip=tip,
                                predicted_score="",
                                scraped_at=now_str
                            )
                            predictions.append(pred)
                            continue

                    # Pattern 2: Full table with percentages and scores
                    if len(tds) >= 8 and "VS" in tds[1]:
                        league_match = tds[1]
                        # e.g. "Argentina: Liga Profesional - ClausuraAtl. TucumanVSArgentinos Jrs"
                        vs_split = re.split(r'VS', league_match, flags=re.IGNORECASE)
                        if len(vs_split) == 2:
                            left = vs_split[0].strip()
                            away = vs_split[1].strip()
                            # left might contain "LeagueNameHomeTeam"
                            # If there's a colon, league is before the colon
                            league = "General"
                            if ":" in left:
                                league_part, home_part = left.split(":", 1)
                                league = league_part.strip()
                                # Clean home part
                                # Remove sub-tournament strings if attached
                                home = re.sub(r'^[A-Za-z\s\-]+(?=[A-Z][a-z])', '', home_part).strip()
                                if not home:
                                    home = home_part.strip()
                            else:
                                home = left

                            # Parse percentages: e.g. '126%X32%242%' -> 1: 26%, X: 32%, 2: 42%
                            prob_str = tds[2]
                            p1, px, p2 = 0, 0, 0
                            pct_matches = re.findall(r'(\d+)%', prob_str)
                            if len(pct_matches) >= 3:
                                try:
                                    p1, px, p2 = int(pct_matches[0]), int(pct_matches[1]), int(pct_matches[2])
                                except ValueError:
                                    pass

                            tip = tds[6].strip() if len(tds) > 6 else ""
                            score = tds[7].strip() if len(tds) > 7 else ""

                            pred = BetensuredPrediction(
                                home_team=home,
                                away_team=away,
                                league=league,
                                prob_home=p1,
                                prob_draw=px,
                                prob_away=p2,
                                tip=tip,
                                predicted_score=score,
                                scraped_at=now_str
                            )
                            predictions.append(pred)

            logger.info(f"Betensured scraped {len(predictions)} match predictions")
        except Exception as e:
            logger.warning(f"Error scraping Betensured: {e}")

        return predictions
