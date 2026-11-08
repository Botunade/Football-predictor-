"""
SportyBet Live Odds Scraper
Extracts real-time, live bookmaker odds directly from SportyBet Nigeria
(1X2, Over/Under, Double Chance) to replace dummy static placeholders.
"""

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from playwright.async_api import async_playwright, Browser, BrowserContext

logger = logging.getLogger(__name__)


@dataclass
class SportyBetFixtureOdds:
    """Live match odds extracted from SportyBet"""
    home_team: str
    away_team: str
    league: str = "Football"
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    match_time: str = ""
    sporty_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SportyBetOddsScraper:
    """
    Scrapes real-time odds from SportyBet Nigeria via Playwright and API endpoints.
    """

    SPORTY_FOOTBALL_URL = "https://www.sportybet.com/ng/sport/football"
    API_PREMATCH_URL = "https://www.sportybet.com/api/ng/factsCenter/liveOrPrematchEvents?sportId=sr%3Asport%3A1&timeline=24"

    def __init__(self, headless: bool = True, timeout: int = 40000):
        self.headless = headless
        self.timeout = timeout
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._playwright = None

    async def __aenter__(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled"
            ]
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass

    async def scrape_live_odds(self) -> List[SportyBetFixtureOdds]:
        """
        Scrapes all active matches and true bookmaker odds from SportyBet football page.
        """
        fixtures: List[SportyBetFixtureOdds] = []

        # 1. Primary: Scrape live DOM
        if self._context:
            try:
                page = await self._context.new_page()
                logger.info(f"Navigating to {self.SPORTY_FOOTBALL_URL} for live odds...")
                await page.goto(self.SPORTY_FOOTBALL_URL, timeout=self.timeout, wait_until="domcontentloaded")
                await page.wait_for_timeout(3500)

                raw_matches = await page.evaluate("""() => {
                    const rows = document.querySelectorAll('.match-row, .m-table-row');
                    const data = [];
                    const seen = new Set();
                    
                    rows.forEach(r => {
                        const teams = r.querySelectorAll('.m-team-name, .home-team, .away-team');
                        if (teams.length < 2) return;
                        
                        const home = teams[0].innerText.trim();
                        const away = teams[1].innerText.trim();
                        if (!home || !away) return;
                        
                        const pairKey = `${home}_vs_${away}`.toLowerCase();
                        if (seen.has(pairKey)) return;
                        seen.add(pairKey);
                        
                        // Extract 1X2 odds
                        const oddsCells = r.querySelectorAll('.m-table-cell--odds, .m-outcome-odds, .af-button--odds');
                        const odds = [];
                        oddsCells.forEach(c => {
                            const val = parseFloat(c.innerText.trim());
                            if (!isNaN(val) && val > 1.0) odds.push(val);
                        });
                        
                        // Time & Match ID
                        let matchTime = "";
                        let sportyId = "";
                        const timeEl = r.querySelector('.m-table-cell--time, .time, .clock');
                        if (timeEl) matchTime = timeEl.innerText.trim();
                        
                        const idEl = r.querySelector('.id, .m-table-cell--id');
                        if (idEl) sportyId = idEl.innerText.replace(/[^0-9]/g, '');
                        
                        // League title from header
                        let league = "Football";
                        const parentTable = r.closest('.m-table');
                        if (parentTable) {
                            const header = parentTable.querySelector('.m-table-header, .league-title, .title');
                            if (header) league = header.innerText.trim();
                        }
                        
                        data.push({
                            home_team: home,
                            away_team: away,
                            league: league,
                            odds_home: odds.length >= 1 ? odds[0] : 0.0,
                            odds_draw: odds.length >= 2 ? odds[1] : 0.0,
                            odds_away: odds.length >= 3 ? odds[2] : 0.0,
                            match_time: matchTime,
                            sporty_id: sportyId
                        });
                    });
                    return data;
                }""")

                await page.close()

                for m in raw_matches:
                    if m.get("odds_home") > 1.0 and m.get("odds_away") > 1.0:
                        fixtures.append(SportyBetFixtureOdds(**m))

                logger.info(f"Successfully scraped {len(fixtures)} live fixtures from SportyBet DOM.")
            except Exception as e:
                logger.warning(f"Error scraping SportyBet DOM: {e}")

        # 2. Fast API supplement/fallback if DOM returned few matches
        if len(fixtures) < 15:
            try:
                api_fixtures = self._scrape_via_api()
                existing_keys = {f"{f.home_team.lower()}_vs_{f.away_team.lower()}" for f in fixtures}
                for af in api_fixtures:
                    k = f"{af.home_team.lower()}_vs_{af.away_team.lower()}"
                    if k not in existing_keys:
                        fixtures.append(af)
                        existing_keys.add(k)
                logger.info(f"Total fixtures after SportyBet API supplement: {len(fixtures)}")
            except Exception as e:
                logger.debug(f"SportyBet API supplement failed: {e}")

        return fixtures

    def _scrape_via_api(self) -> List[SportyBetFixtureOdds]:
        """HTTP API fallback for SportyBet live & prematch fixtures"""
        results: List[SportyBetFixtureOdds] = []
        req = urllib.request.Request(
            self.API_PREMATCH_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))

        for group in data.get("data", []):
            league_name = group.get("categoryName") or group.get("tournamentName") or "Football"
            for ev in group.get("events", []):
                home = ev.get("homeTeamName")
                away = ev.get("awayTeamName")
                if not home or not away:
                    continue

                h_odd, d_odd, a_odd = 0.0, 0.0, 0.0
                for m in ev.get("markets", []):
                    desc = m.get("desc", "").lower()
                    if "1x2" in desc or m.get("id") == "1":
                        for o in m.get("outcomes", []):
                            d_desc = o.get("desc", "")
                            val = float(o.get("odds", 0.0))
                            if d_desc == "1" or "home" in d_desc.lower():
                                h_odd = val
                            elif d_desc == "X" or "draw" in d_desc.lower():
                                d_odd = val
                            elif d_desc == "2" or "away" in d_desc.lower():
                                a_odd = val

                if h_odd > 1.0 and a_odd > 1.0:
                    results.append(SportyBetFixtureOdds(
                        home_team=home,
                        away_team=away,
                        league=league_name,
                        odds_home=h_odd,
                        odds_draw=d_odd,
                        odds_away=a_odd,
                        sporty_id=str(ev.get("eventId", ""))
                    ))

        return results
