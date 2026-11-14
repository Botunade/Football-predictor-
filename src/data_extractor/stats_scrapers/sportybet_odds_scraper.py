"""
SportyBet Live Odds Scraper
Extracts real-time, live bookmaker odds directly from SportyBet Nigeria
(1X2, Over/Under, Double Chance) to replace dummy static placeholders.
"""

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple

from playwright.async_api import async_playwright, Browser, BrowserContext

try:
    from ..code_reader import normalize_sportybet_selection
except (ImportError, ValueError):
    try:
        from data_extractor.code_reader import normalize_sportybet_selection
    except ImportError:
        try:
            from src.data_extractor.code_reader import normalize_sportybet_selection
        except ImportError:
            normalize_sportybet_selection = None

logger = logging.getLogger(__name__)


@dataclass
class SportyBetFixtureOdds:
    """Live match odds extracted from SportyBet (1X2 + Multi-Market)"""
    home_team: str
    away_team: str
    league: str = "Football"
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0
    match_time: str = ""
    sporty_id: str = ""
    start_time_ts: int = 0
    market_odds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SportyBetOddsScraper:
    """
    Scrapes real-time odds from SportyBet Nigeria via Playwright and API endpoints.
    """

    SPORTY_FOOTBALL_URL = "https://www.sportybet.com/ng/sport/football"
    UPCOMING_EVENTS_BASE = (
        "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents?"
        "sportId=sr%3Asport%3A1&"
        "marketId=1%2C18%2C10%2C29%2C11%2C26%2C19%2C20%2C31%2C32%2C50%2C51%2C60%2C63%2C64%2C68%2C75%2C83%2C85%2C86%2C90%2C95%2C548%2C60100%2C60200&"
        "pageSize=100&option=1"
    )

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

        # 2. Comprehensive API supplement (fetches 500+ matches across upcoming & live tournaments)
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

    @staticmethod
    def _parse_event_markets(ev: Dict[str, Any]) -> Tuple[float, float, float, Dict[str, float]]:
        """Extracts 1X2 and 45+ multi-market odds (1st Half, 2nd Half, DC, O/U, GG/NG, DNB, Team O/U, 1UP/2UP) from a SportyBet event payload."""
        h_odd, d_odd, a_odd = 0.0, 0.0, 0.0
        m_odds: Dict[str, float] = {}

        for m in ev.get("markets", []):
            m_id = str(m.get("id", "")).strip()
            m_desc = str(m.get("desc", "")).strip()
            spec = str(m.get("specifier") or "").strip()
            outcomes = m.get("outcomes", [])

            for o in outcomes:
                o_id = str(o.get("id", "")).strip()
                o_desc = str(o.get("desc", "")).strip()
                try:
                    val = float(o.get("odds") or 0.0)
                except (ValueError, TypeError):
                    val = 0.0
                if val <= 1.0:
                    continue

                if normalize_sportybet_selection is not None:
                    norm_key = normalize_sportybet_selection(m_desc, o_desc, m_id, spec)
                    if norm_key:
                        m_odds[norm_key] = val

                if m_id == "1":
                    if o_id == "1" or o_desc.lower() == "home":
                        h_odd = val
                        m_odds["Home"] = val
                    elif o_id == "2" or o_desc.lower() == "draw":
                        d_odd = val
                        m_odds["Draw"] = val
                    elif o_id == "3" or o_desc.lower() == "away":
                        a_odd = val
                        m_odds["Away"] = val
                elif m_id == "10":
                    if o_id == "9":
                        m_odds["1X"] = val
                    elif o_id == "10":
                        m_odds["12"] = val
                    elif o_id == "11":
                        m_odds["X2"] = val
                elif m_id == "18" and spec.startswith("total="):
                    line = spec.split("=")[1]
                    if o_id == "12":
                        m_odds[f"Over {line}"] = val
                    elif o_id == "13":
                        m_odds[f"Under {line}"] = val
                elif m_id == "68" and spec.startswith("total="):
                    line = spec.split("=")[1]
                    if o_id == "12":
                        m_odds[f"1st Half Over {line}"] = val
                    elif o_id == "13":
                        m_odds[f"1st Half Under {line}"] = val
                elif m_id == "90" and spec.startswith("total="):
                    line = spec.split("=")[1]
                    if o_id == "12":
                        m_odds[f"2nd Half Over {line}"] = val
                    elif o_id == "13":
                        m_odds[f"2nd Half Under {line}"] = val
                elif m_id == "29":
                    if o_id == "74":
                        m_odds["BTTS Yes"] = val
                    elif o_id == "76":
                        m_odds["BTTS No"] = val
                elif m_id == "11":
                    if o_id == "4":
                        m_odds["Home DNB"] = val
                    elif o_id == "5":
                        m_odds["Away DNB"] = val

        return h_odd, d_odd, a_odd, m_odds

    def _scrape_via_api(self) -> List[SportyBetFixtureOdds]:
        """
        High-scale SportyBet official API extraction.
        Queries multiple pages of upcoming events and live prematch events,
        yielding 500+ real 1X2 and Multi-Market bookmaker odds with exact event IDs.
        """
        import time
        from datetime import datetime
        from concurrent.futures import ThreadPoolExecutor

        results: List[SportyBetFixtureOdds] = []
        seen_keys = set()
        timestamp = int(time.time() * 1000)
        now_sec = time.time()
        now_dt = datetime.now()

        def _format_match_time(est_start_ms: int) -> Tuple[str, int]:
            if not est_start_ms or est_start_ms <= 0:
                return "", 0
            ts_sec = int(est_start_ms / 1000)
            try:
                dt = datetime.fromtimestamp(ts_sec)
                if dt.date() == now_dt.date():
                    return f"Today {dt.strftime('%H:%M')}", ts_sec
                elif (dt.date() - now_dt.date()).days == 1:
                    return f"Tomorrow {dt.strftime('%H:%M')}", ts_sec
                else:
                    return dt.strftime("%b %d %H:%M"), ts_sec
            except Exception:
                return "", ts_sec

        # Construct comprehensive endpoint list across major and lower leagues:
        # 1. Live and imminent pre-match events
        urls = [
            f"https://www.sportybet.com/api/ng/factsCenter/liveOrPrematchEvents?sportId=sr%3Asport%3A1&timeline=24&_t={timestamp}"
        ]

        # 2. Upcoming matches today (timeline=24, pages 1 to 7)
        for p in range(1, 8):
            urls.append(f"{self.UPCOMING_EVENTS_BASE}&timeline=24&pageNum={p}&_t={timestamp}")

        # 3. Upcoming matches next 48h (timeline=48, pages 1 to 5)
        for p in range(1, 6):
            urls.append(f"{self.UPCOMING_EVENTS_BASE}&timeline=48&pageNum={p}&_t={timestamp}")

        # 4. General lower leagues & regional tournaments (pages 1 to 6)
        for p in range(1, 7):
            urls.append(f"{self.UPCOMING_EVENTS_BASE}&pageNum={p}&_t={timestamp}")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "application/json"
        }

        def fetch_payload(url: str) -> Optional[Dict[str, Any]]:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=8) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception:
                return None

        # Fetch all pages concurrently with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as executor:
            payloads = list(executor.map(fetch_payload, urls))

        for data in payloads:
            if not data:
                continue
            raw_data = data.get("data")
            groups = raw_data if isinstance(raw_data, list) else (raw_data.get("tournaments", []) if isinstance(raw_data, dict) else [])

            for group in groups:
                cat = (group.get("categoryName") or "").strip()
                t_name = (group.get("name") or group.get("tournamentName") or "Football").strip()
                if cat and cat.lower() not in t_name.lower() and cat not in ["International", "World"]:
                    league_name = f"{cat} - {t_name}"
                else:
                    league_name = t_name

                for ev in group.get("events", []):
                    home = ev.get("homeTeamName")
                    away = ev.get("awayTeamName")
                    if not home or not away:
                        continue
                    pair_key = f"{home.lower()}_vs_{away.lower()}"
                    if pair_key in seen_keys:
                        continue

                    est_start = int(ev.get("estimateStartTime") or 0)
                    m_time_str, st_ts = _format_match_time(est_start)

                    # Exclude matches that started >30m ago
                    if st_ts > 0 and st_ts < (now_sec - 1800):
                        continue

                    h_odd, d_odd, a_odd, m_odds = self._parse_event_markets(ev)

                    if h_odd > 1.0 and a_odd > 1.0:
                        seen_keys.add(pair_key)
                        results.append(SportyBetFixtureOdds(
                            home_team=home,
                            away_team=away,
                            league=league_name,
                            odds_home=h_odd,
                            odds_draw=d_odd,
                            odds_away=a_odd,
                            match_time=m_time_str,
                            sporty_id=str(ev.get("eventId", "")),
                            start_time_ts=st_ts,
                            market_odds=m_odds
                        ))

        # Sort chronologically by start time: closest upcoming fixtures first!
        results.sort(key=lambda x: (0 if x.start_time_ts > 0 else 1, x.start_time_ts if x.start_time_ts > 0 else 9999999999))

        logger.info(f"SportyBet high-scale API extraction completed: {len(results)} total fixtures across major & lower leagues.")
        return results
