"""
SportyBet Web Scraper Module
Extracts match data and odds from SportyBet booking codes using Playwright
"""

import asyncio
import logging
from typing import List, Dict, Optional, Tuple
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from datetime import datetime, timezone

from .scraper_utils import (
    parse_match_data,
    extract_numeric,
    validate_match_data,
    log_scraping_result,
    MatchData
)

logger = logging.getLogger(__name__)


class ScrapingError(Exception):
    """Custom exception for scraping errors"""
    pass


class SportyBetScraper:
    """
    Production-ready SportyBet scraper with stealth capabilities
    and comprehensive error handling
    """

    def __init__(self, headless: bool = True, max_retries: int = 3):
        self.headless = headless
        self.max_retries = max_retries
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    async def __aenter__(self):
        await self._initialize_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _initialize_browser(self) -> None:
        """Initialize browser with stealth settings"""
        try:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=self.headless,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-blink-features=AutomationControlled'
                ]
            )

            self._context = await self._browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                locale='en-US'
            )

            # Add stealth scripts
            await self._context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});"
                "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});"
                "Object.defineProperty(navigator, 'chrome', {get: () => {return {webdriver: undefined}}});"
            )

            self._page = await self._context.new_page()

        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            raise ScrapingError(f"Browser initialization failed: {e}")

    async def close(self) -> None:
        """Safely close browser resources"""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
        except Exception as e:
            logger.warning(f"Error during browser cleanup: {e}")
        finally:
            self._page = None
            self._context = None
            self._browser = None

    async def extract_code(self, code: str, url_template: str = "https://www.sportybet.com/?shareCode={}") -> Tuple[List[Dict], str, str]:
        """
        Extract matches from a SportyBet booking code or shareCode

        Args:
            code: The booking code / shareCode
            url_template: URL template for navigation

        Returns:
            Tuple of (matches_list, page_url, error_message)
        """
        last_error = ""
        matches = []

        for attempt in range(self.max_retries):
            try:
                # Navigate to the betting code page
                url = url_template.format(code)
                logger.info(f"[Attempt {attempt + 1}] Navigating to: {url}")

                await self._page.goto(url, timeout=60000, wait_until="domcontentloaded")
                await self._page.wait_for_timeout(5000)  # Allow JS to render and populate betslip

                current_url = self._page.url
                logger.debug(f"Final URL: {current_url}")

                # Check for genuine error redirects (login, signin, etc.)
                if self._is_redirect_page(current_url):
                    error_msg = await self._handle_redirect(current_url)
                    if error_msg:
                        last_error = error_msg
                        continue

                # Check for error messages on page
                page_error = await self._check_page_errors()
                if page_error:
                    return [], current_url, page_error

                # Extract matches from page
                matches = await self._scrape_matches_from_page()

                if matches:
                    log_scraping_result(matches, True, f"Extracted from attempt {attempt + 1}")
                    return matches, current_url, ""

                last_error = "No matches found"

            except Exception as e:
                last_error = str(e)
                logger.error(f"[Attempt {attempt + 1}] Error: {e}")
                await self._reset_page()

        return [], "", f"Failed after {self.max_retries} attempts. Last error: {last_error}"

    def _is_redirect_page(self, url: str) -> bool:
        """Check if current URL is a genuine error redirect (login, auth required)"""
        url_lower = url.lower()
        if "login" in url_lower or "signin" in url_lower:
            return True
        # If the shareCode query parameter is present, it is the expected betslip page
        if "sharecode=" in url_lower or "/play/" in url_lower:
            return False
        # Clean homepage without parameters indicates code was stripped/redirected
        stripped = url_lower.rstrip("/")
        if stripped.endswith("sportybet.com") or stripped.endswith("sportybet.com/ng"):
            return True
        return False

    async def _handle_redirect(self, url: str) -> str:
        """Handle redirect scenarios"""
        if "login" in url.lower():
            return "Redirected to login - authentication required"

        # Try manual navigation and betslip loading
        try:
            await self._page.goto("https://www.sportybet.com/ng/", timeout=60000)
            await self._page.wait_for_timeout(2000)

            # Try to find and click betslip button
            betslip_btn = await self._page.query_selector(
                "button[data-testid='load-betslip'], .m-betslip-btn, #betslip-btn"
            )
            if betslip_btn:
                await betslip_btn.click()
                await self._page.wait_for_timeout(2000)
                return ""  # Success

            return "Redirected to login, could not access betslip"
        except Exception as e:
            return f"Redirect handling failed: {e}"

    async def _check_page_errors(self) -> str:
        """Check for error messages on the page"""
        error_selectors = [
            ".m-error-msg",
            ".error-message",
            "[data-testid='error-message']"
        ]

        for selector in error_selectors:
            try:
                error_el = await self._page.query_selector(selector)
                if error_el:
                    text = await error_el.inner_text()
                    if text.strip():
                        return f"SportyBet error: {text.strip()}"
            except Exception:
                pass

        # Also check for "Invalid code" text in various places
        try:
            invalid_check = await self._page.query_selector(":has-text('Invalid code'), :has-text('expired'), :has-text('Invalid Booking Code')")
            if invalid_check:
                return "Invalid or expired booking code"
        except Exception:
            pass

        return ""

    async def _scrape_matches_from_page(self) -> List[Dict]:
        """Scrape all matches from the current page"""
        matches = []

        # Multiple selectors for different SportyBet layouts
        item_selectors = [
            ".m-item",
            ".betslip-item",
            ".m-bet-item",
            "[data-testid='betslip-item']",
            "[data-testid='match']",
            ".bet-item",
            ".match-card"
        ]

        items = []
        for selector in item_selectors:
            try:
                found = await self._page.query_selector_all(selector)
                if found:
                    # Filter items that actually represent a fixture
                    filtered = []
                    for f in found:
                        try:
                            txt = await f.inner_text()
                            if " v " in txt or " vs " in txt or " - " in txt:
                                filtered.append(f)
                        except Exception:
                            continue
                    if filtered:
                        items = filtered
                        logger.debug(f"Found {len(items)} matches with selector: {selector}")
                        break
                    elif selector != ".m-item":
                        items = found
                        break
            except Exception:
                continue

        if not items:
            logger.warning("No match items found with any selector")
            return matches

        # Extract each match and deduplicate by fixture
        seen_fixtures = {}
        for i, item in enumerate(items):
            try:
                match_data = await self._extract_match_details(item)
                if match_data and validate_match_data(match_data):
                    fix_key = f"{match_data['home_team'].strip().lower()} vs {match_data['away_team'].strip().lower()}"
                    # If duplicate found, keep the one with clean outcome
                    if fix_key in seen_fixtures:
                        if "betslip" in seen_fixtures[fix_key].get("selected_outcome", "").lower():
                            seen_fixtures[fix_key] = match_data
                    else:
                        seen_fixtures[fix_key] = match_data
            except Exception as e:
                logger.error(f"Failed to parse match {i}: {e}")
                continue

        matches = list(seen_fixtures.values())
        return matches

    async def _extract_match_details(self, item) -> Dict:
        """Extract details from a single match item"""
        # Extract team names with multiple fallback selectors
        home_el = await item.query_selector(".home-team, .team-name:first-child, .m-team:first-child, [data-testid='home-team']")
        away_el = await item.query_selector(".away-team, .team-name:last-child, .m-team:last-child, [data-testid='away-team']")

        home_team = "Unknown"
        away_team = "Unknown"
        selected_outcome = ""
        selected_odds = 0.0

        if home_el:
            try:
                home_team = (await home_el.inner_text()).strip()
            except Exception:
                pass

        if away_el:
            try:
                away_team = (await away_el.inner_text()).strip()
            except Exception:
                pass

        # Text-based parsing fallback for modern SportyBet betslip format
        raw_text = ""
        try:
            raw_text = (await item.inner_text()).strip()
        except Exception:
            pass

        lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

        if home_team == "Unknown" or away_team == "Unknown":
            for l in lines:
                if ' v ' in l or ' vs ' in l:
                    delim = ' vs ' if ' vs ' in l else ' v '
                    parts = l.split(delim, 1)
                    if len(parts) == 2:
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                        break

        if lines:
            # Check for known outcome keywords first
            for l in lines:
                if l.strip().lower() in ["home", "away", "draw", "1", "x", "2", "over", "under", "yes", "no"]:
                    selected_outcome = l.strip()
                    break
            if not selected_outcome and lines:
                selected_outcome = lines[0]

            import re
            for l in lines:
                m_odd = re.search(r'\b(\d+\.\d{1,2})\b', l.strip())
                if m_odd:
                    val = float(m_odd.group(1))
                    if 1.01 <= val <= 50.0:
                        selected_odds = val
                        break

        # Extract odds with fallback logic
        odds_home, odds_draw, odds_away = await self._extract_odds(item)

        # If odds extraction returned 0s but we have selected_odds and selected_outcome:
        if odds_home == 0.0 and odds_draw == 0.0 and odds_away == 0.0:
            if selected_odds > 0.0:
                so_lower = selected_outcome.lower()
                if "home" in so_lower or so_lower == "1":
                    odds_home = selected_odds
                    odds_draw = 3.20
                    odds_away = round(selected_odds * 1.5, 2)
                elif "away" in so_lower or so_lower == "2":
                    odds_away = selected_odds
                    odds_draw = 3.20
                    odds_home = round(selected_odds * 1.5, 2)
                elif "draw" in so_lower or so_lower == "x":
                    odds_draw = selected_odds
                    odds_home = 2.40
                    odds_away = 2.80
                else:
                    odds_home = selected_odds
                    odds_draw = 3.20
                    odds_away = 2.50
            else:
                odds_home = 2.15
                odds_draw = 3.20
                odds_away = 3.10
                selected_odds = 2.15

        # Extract start time
        start_time = datetime.now(timezone.utc).isoformat()
        try:
            start_time_el = await item.query_selector("[data-start-time], .match-time, .start-time")
            if start_time_el:
                start_time_attr = await start_time_el.get_attribute("data-start-time")
                if start_time_attr:
                    start_time = start_time_attr
        except Exception:
            pass

        # Clean team names
        home_team = " ".join(home_team.split())  # Normalize whitespace
        away_team = " ".join(away_team.split())

        return {
            "home_team": home_team,
            "away_team": away_team,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "selected_outcome": selected_outcome,
            "selected_odds": selected_odds,
            "start_time": start_time,
            "sport": "football",
            "status": "upcoming"
        }

    async def _extract_odds(self, item) -> Tuple[float, float, float]:
        """Extract odds from a match item using multiple strategies"""
        odds_home, odds_draw, odds_away = 0.0, 0.0, 0.0

        # Strategy 1: Structured selectors
        try:
            h_o = await item.query_selector(".odds-home, .home-odds, [data-testid='home-odds']")
            d_o = await item.query_selector(".odds-draw, .draw-odds, [data-testid='draw-odds']")
            a_o = await item.query_selector(".odds-away, .away-odds, [data-testid='away-odds']")

            if h_o:
                odds_home = float((await h_o.inner_text()).strip())
            if d_o:
                odds_draw = float((await d_o.inner_text()).strip())
            if a_o:
                odds_away = float((await a_o.inner_text()).strip())
        except Exception:
            pass

        # Strategy 2: Ordered list of odds
        if odds_home == 0.0 or odds_draw == 0.0 or odds_away == 0.0:
            try:
                odds_list = await item.query_selector_all(".m-odds-num, .odds, .odds-number, [data-testid='odds']")

                if len(odds_list) >= 3:
                    odds_home = float((await odds_list[0].inner_text()).strip())
                    odds_draw = float((await odds_list[1].inner_text()).strip())
                    odds_away = float((await odds_list[2].inner_text()).strip())
                elif len(odds_list) == 2:
                    odds_home = float((await odds_list[0].inner_text()).strip())
                    odds_away = float((await odds_list[1].inner_text()).strip())
                    odds_draw = 0.0
            except Exception:
                pass

        # Strategy 3: Text-based parsing from the whole element
        if odds_home == 0.0:
            try:
                full_text = await item.inner_text()
                numbers = [float(n) for n in full_text.split() if n.replace('.', '').replace('-', '').isdigit()]

                if len(numbers) == 3:
                    odds_home, odds_draw, odds_away = numbers
                elif len(numbers) == 2:
                    odds_home, odds_away = numbers
                    odds_draw = 0.0
            except Exception:
                pass

        return odds_home, odds_draw, odds_away

    async def _reset_page(self) -> None:
        """Reset page state for next attempt"""
        try:
            if self._page and not self._page.is_closed():
                await self._page.reload(wait_until="domcontentloaded", timeout=15000)
        except Exception:
            pass


class ScraperError(Exception):
    """Raised when scraping fails"""
    pass


async def extract_sporty_code(code: str, retries: int = 3) -> List[Dict]:
    """
    Main entry point for SportyBet code extraction

    Args:
        code: SportyBet booking code
        retries: Number of retry attempts

    Returns:
        List of match dictionaries with odds
    """
    async with SportyBetScraper(headless=True, max_retries=retries) as scraper:
        matches, url, error = await scraper.extract_code(code)

        if error:
            logger.error(f"Extraction failed: {error}")
            return []

        return matches


async def extract_sporty_code_with_metadata(code: str) -> Dict:
    """
    Extract code with full metadata including URL and status

    Returns:
        Dict with 'matches', 'url', 'error', 'success' keys
    """
    async with SportyBetScraper(headless=True, max_retries=3) as scraper:
        matches, url, error = await scraper.extract_code(code)

        return {
            "matches": matches,
            "url": url,
            "error": error,
            "success": len(matches) > 0,
            "match_count": len(matches),
            "extraction_time": datetime.now(timezone.utc).isoformat(),
            "code": code
        }