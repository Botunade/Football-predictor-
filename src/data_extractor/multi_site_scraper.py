"""
Multi-Site SportyBet Scraper Module
Aggressive multi-site scraping with stealth capabilities for various bookmakers

Supports:
- SportyBet Nigeria
- SportyBet UK
- Bet9ja
- Nairabet
- Betinng
- WinWin
- And other regional bookmakers
"""

import asyncio
import json
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Callable
from pathlib import Path

from playwright.async_api import (
    async_playwright,
    Page,
    Browser,
    BrowserContext,
    Error as PlaywrightError,
    Request as PlaywrightRequest
)

logger = logging.getLogger(__name__)


class Bookmaker(Enum):
    """Supported bookmakers"""
    SPORTYBET_NG = "sportybet_ng"
    SPORTYBET_UK = "sportybet_uk"
    BET9JA = "bet9ja"
    NAIRABET = "nairabet"
    BETINNG = "betinng"
    WINWIN = "winwin"
    CUSTOM = "custom"


@dataclass
class ScraperConfig:
    """Configuration for multi-site scraper with proxy and rate limiting support"""
    # Basic settings
    headless: bool = True
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 30.0
    retry_exponential_base: float = 2.0

    # Proxy settings
    proxy_host: Optional[str] = None
    proxy_port: Optional[int] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    proxy_enabled: bool = False

    # Rate limiting
    min_delay: float = 1.0
    max_delay: float = 5.0
    request_timeout: int = 60000

    # User agent rotation
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    ])

    # Viewport sizes for randomization
    viewport_sizes: List[Dict[str, int]] = field(default_factory=lambda: [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720},
        {"width": 375, "height": 812},  # Mobile
    ])

    # Locale settings
    locales: List[str] = field(default_factory=lambda: ["en-US", "en-GB", "en-Nigeria"])

    # Session persistence
    session_storage_path: Optional[str] = None

    # Output settings
    output_include_raw_html: bool = False
    output_directory: str = "scraped_data"

    # Stealth settings
    stealth_enabled: bool = True
    random_move_delay: int = 1000  # ms

    # Legacy sites configuration (for backward compatibility)
    sites: Dict[str, str] = field(default_factory=lambda: {
        "sportybet_ng": "https://www.sportybet.com/ng/play/{code}",
        "sportybet_uk": "https://www.sportybet.com/en/play/{code}",
        "sportybet_ie": "https://www.sportybet.com/ie/play/{code}",
    })

    def __post_init__(self):
        """Validate and normalize configuration"""
        if not self.user_agents:
            self.user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ]

    @property
    def timeout(self) -> int:
        """Alias for request_timeout for backward compatibility"""
        return self.request_timeout


@dataclass
class ScraperSiteConfig:
    """Site-specific configuration for different bookmakers"""
    name: str
    base_url: str
    code_param: str = "code"
    betslip_selectors: List[str] = field(default_factory=lambda: [".betslip", "#betslip", ".m-betslip-btn"])
    match_selectors: List[str] = field(default_factory=lambda: [
        ".betslip-item", ".m-bet-item", "[data-testid='betslip-item']", ".match-card", ".bet-item"
    ])
    team_name_selectors: Dict[str, List[str]] = field(default_factory=lambda: {
        "home": [".home-team", ".team-name:first-child", ".m-team:first-child"],
        "away": [".away-team", ".team-name:last-child", ".m-team:last-child"]
    })
    odds_selectors: Dict[str, List[str]] = field(default_factory=lambda: {
        "home": [".odds-home", ".home-odds", "[data-testid='home-odds']"],
        "draw": [".odds-draw", ".draw-odds", "[data-testid='draw-odds']"],
        "away": [".odds-away", ".away-odds", "[data-testid='away-odds']"]
    })
    auth_required: bool = False
    login_url: Optional[str] = None


# Predefined site configurations
SITE_CONFIGS = {
    Bookmaker.SPORTYBET_NG: ScraperSiteConfig(
        name="SportyBet Nigeria",
        base_url="https://www.sportybet.com/ng",
        code_param="",
        match_selectors=[".betslip-item", ".m-bet-item", "[data-testid='betslip-item']"]
    ),
    Bookmaker.SPORTYBET_UK: ScraperSiteConfig(
        name="SportyBet UK",
        base_url="https://www.sportybet.com",
        code_param=""
    ),
    Bookmaker.BET9JA: ScraperSiteConfig(
        name="Bet9ja",
        base_url="https://www.bet9ja.com",
        code_param="b",
        match_selectors=[".bet-item", ".match-card"]
    ),
    Bookmaker.NAIRABET: ScraperSiteConfig(
        name="Nairabet",
        base_url="https://www.nairabet.com",
        code_param=""
    ),
    Bookmaker.BETINNG: ScraperSiteConfig(
        name="Betinng",
        base_url="https://www.betinng.com",
        code_param=""
    ),
    Bookmaker.WINWIN: ScraperSiteConfig(
        name="WinWin",
        base_url="https://www.winwinbet.com",
        code_param=""
    ),
    Bookmaker.CUSTOM: ScraperSiteConfig(
        name="Custom Site",
        base_url="https://example.com"
    )
}


@dataclass
class ScrapeResult:
    """Result of a scraping operation with JSON serialization"""
    success: bool
    matches: List[Dict[str, Any]]
    source_url: str
    source_site: str
    error_message: str
    timestamp: str
    scraper_version: str = "2.0.0"
    request_count: int = 0
    extraction_duration_ms: float = 0.0
    raw_html: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    documentation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "success": self.success,
            "matches": self.matches,
            "source_url": self.source_url,
            "source_site": self.source_site,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "scraper_version": self.scraper_version,
            "request_count": self.request_count,
            "extraction_duration_ms": self.extraction_duration_ms,
            "metadata": self.metadata
        }
        if self.raw_html:
            result["raw_html"] = self.raw_html
        if self.documentation:
            result["documentation"] = self.documentation
        return result

    def to_json(self, include_raw_html: bool = False) -> str:
        """Convert to JSON string with optional raw HTML"""
        data = self.to_dict()
        if not include_raw_html and "raw_html" in data:
            del data["raw_html"]
        return json.dumps(data, indent=2, default=str)

    def validate_schema(self) -> bool:
        """Validate that result matches expected schema"""
        required_fields = ["success", "matches", "source_url", "source_site", "timestamp"]
        for field_name in required_fields:
            if field_name not in self.to_dict():
                return False
        return True


class ScrapingError(Exception):
    """Custom exception for scraping errors"""
    pass


class MultiSiteScraper:
    """
    Multi-site web scraper with aggressive capabilities for sports betting sites.

    Features:
    - Proxy support with full authentication
    - User agent rotation
    - Exponential backoff retry logic
    - Session persistence across requests
    - Stealth techniques with JavaScript injection
    - Request interception for cookie management
    - Randomized viewport sizes
    - Context manager support

    Supported bookmakers:
    - SportyBet Nigeria
    - SportyBet UK
    - Bet9ja
    - Nairabet
    - Betinng
    - WinWin
    - Custom sites
    """

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        site_config: Optional[ScraperSiteConfig] = None
    ):
        self.config = config or ScraperConfig()
        self.site_config = site_config or SITE_CONFIGS[Bookmaker.CUSTOM]
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._request_count = 0
        self._start_time: float = 0
        self._current_user_agent: str = ""
        self._current_viewport: Dict[str, int] = {}
        self._session_id: str = str(uuid.uuid4())

    async def __aenter__(self) -> "MultiSiteScraper":
        """Async context manager entry - initializes browser with stealth settings"""
        await self._initialize_browser()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - safely closes browser resources"""
        await self.close()

    async def _initialize_browser(self) -> None:
        """Initialize browser with stealth settings and optional proxy"""
        playwright = await async_playwright().start()

        # Select random configuration
        self._current_user_agent = random.choice(self.config.user_agents)
        self._current_viewport = random.choice(self.config.viewport_sizes)

        # Build browser launch arguments
        launch_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--start-maximized",
            "--disable-extensions",
            "--disable-plugin-loading",
            "--disable-component-extension-optional-loading",
        ]

        browser_kwargs = {
            "headless": self.config.headless,
            "args": launch_args
        }

        # Proxy configuration
        if self.config.proxy_enabled and self.config.proxy_host and self.config.proxy_port:
            browser_kwargs["proxy"] = {
                "server": f"http://{self.config.proxy_host}:{self.config.proxy_port}",
                "username": self.config.proxy_username,
                "password": self.config.proxy_password
            }

        try:
            self._browser = await playwright.chromium.launch(**browser_kwargs)
            await self._create_stealth_context()
            logger.info(f"Browser initialized for {self.site_config.name}")
        except PlaywrightError as e:
            logger.error(f"Failed to launch browser: {e}")
            raise ScrapingError(f"Browser launch failed: {e}")

    async def _create_stealth_context(self) -> None:
        """Create browser context with stealth settings"""
        locale = random.choice(self.config.locales)

        context_args = {
            "viewport": self._current_viewport,
            "user_agent": self._current_user_agent,
            "locale": locale,
            "timezone_id": "UTC",
            "geolocation": {"latitude": 0, "longitude": 0},
            "permissions": []
        }

        if self.config.proxy_enabled and self.config.proxy_host and self.config.proxy_port:
            context_args["proxy"] = {
                "server": f"http://{self.config.proxy_host}:{self.config.proxy_port}",
                "username": self.config.proxy_username,
                "password": self.config.proxy_password
            }

        self._context = await self._browser.new_context(**context_args)

        # Add stealth scripts
        await self._add_stealth_scripts()

        # Create page
        self._page = await self._context.new_page()

        # Set up request interception for cookie management
        await self._page.route("**/*", self._intercept_request)

    async def _add_stealth_scripts(self) -> None:
        """Inject JavaScript to appear more human-like"""
        if not self.config.stealth_enabled:
            return

        # Hide webdriver property
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        # Fake plugins
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """)

        # Fake languages
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        """)

        # Fake Chrome properties
        await self._context.add_init_script("""
            Object.defineProperty(navigator, 'chrome', {
                get: () => ({
                    webdriver: undefined,
                    _disableTest: false,
                    runtime: {}
                })
            });
        """)

        # Fake screen properties
        await self._context.add_init_script("""
            Object.defineProperty(screen, 'colorDepth', {get: () => 24});
            Object.defineProperty(screen, 'pixelDepth', {get: () => 24});
            Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
        """)

        logger.debug("Stealth scripts injected")

    async def _intercept_request(self, route, request: PlaywrightRequest) -> None:
        """Intercept requests for cookie management and logging"""
        self._request_count += 1

        url = request.url
        if "betslip" in url.lower() or "match" in url.lower():
            logger.debug(f"[Request {self._request_count}] {url}")

        await route.continue_()

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

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff including jitter"""
        base_delay = self.config.retry_delay
        exponential_delay = base_delay * (self.config.retry_exponential_base ** attempt)
        jitter = random.uniform(0, 1)
        return min(exponential_delay + jitter, self.config.max_retry_delay)

    async def extract_from_url(
        self,
        url: str,
        betting_code: Optional[str] = None
    ) -> ScrapeResult:
        """
        Extract data from a URL with a betting code

        Args:
            url: Target URL
            betting_code: Optional betting code to append

        Returns:
            ScrapeResult with extracted data
        """
        self._start_time = time.time()
        self._request_count = 0
        matches: List[Dict[str, Any]] = []
        raw_html: Optional[str] = None
        error_message = ""

        # Build full URL
        if betting_code:
            separator = "&" if "?" in url else "?"
            code_param = self.site_config.code_param or "code"
            full_url = f"{url}{separator}{code_param}={betting_code}"
        else:
            full_url = url

        try:
            logger.info(f"Navigating to: {full_url}")

            await self._page.goto(full_url, timeout=self.config.request_timeout, wait_until="networkidle")
            await self._page.wait_for_timeout(3000)

            current_url = self._page.url
            logger.debug(f"Final URL: {current_url}")

            # Check for redirect pages
            if self._is_redirect_page(current_url):
                error_message = await self._handle_redirect(current_url)
                if error_message:
                    return ScrapeResult(
                        success=False,
                        matches=[],
                        source_url=full_url,
                        source_site=self.site_config.name,
                        error_message=error_message,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        request_count=self._request_count,
                        extraction_duration_ms=(time.time() - self._start_time) * 1000
                    )

            # Check for page errors
            error_message = await self._check_page_errors()
            if error_message:
                return ScrapeResult(
                    success=False,
                    matches=[],
                    source_url=current_url,
                    source_site=self.site_config.name,
                    error_message=error_message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    request_count=self._request_count,
                    extraction_duration_ms=(time.time() - self._start_time) * 1000
                )

            if self.config.output_include_raw_html:
                raw_html = await self._page.content()

            matches = await self._scrape_matches_from_page()

            success = len(matches) > 0
            error_message = "" if success else "No matches extracted"

            log_scraping_result(matches, success, f"from {self.site_config.name}")

        except Exception as e:
            error_message = str(e)
            logger.error(f"Extraction failed: {e}")

        duration_ms = (time.time() - self._start_time) * 1000

        return ScrapeResult(
            success=len(matches) > 0,
            matches=matches,
            source_url=full_url,
            source_site=self.site_config.name,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            scraper_version="2.0.0",
            request_count=self._request_count,
            extraction_duration_ms=duration_ms,
            raw_html=raw_html if self.config.output_include_raw_html else None,
            metadata={
                "user_agent": self._current_user_agent,
                "viewport": self._current_viewport,
                "session_id": self._session_id,
                "retry_count": self.config.max_retries
            }
        )

    def _is_redirect_page(self, url: str) -> bool:
        """Check if current URL is a redirect (login, homepage, etc.)"""
        redirect_patterns = ["/login", "/signin", "/account", "login=", "?error=", "error=login"]
        return any(pattern in url.lower() for pattern in redirect_patterns)

    async def _handle_redirect(self, url: str) -> str:
        """Handle redirect scenarios"""
        if "login" in url.lower():
            return "Redirected to login - authentication required"

        try:
            await self._page.goto(self.site_config.base_url, timeout=self.config.request_timeout)
            await self._page.wait_for_timeout(2000)

            for selector in self.site_config.betslip_selectors:
                try:
                    betslip_btn = await self._page.query_selector(selector)
                    if betslip_btn:
                        await betslip_btn.click()
                        await self._page.wait_for_timeout(2000)
                        return ""
                except Exception:
                    continue

            return "Redirected, could not access betslip"
        except Exception as e:
            return f"Redirect handling failed: {e}"

    async def _check_page_errors(self) -> str:
        """Check for error messages on the page"""
        error_selectors = [
            ".error-message", ".m-error-msg", "[data-testid='error-message']",
            ".alert-error", ".error"
        ]

        for selector in error_selectors:
            try:
                error_el = await self._page.query_selector(selector)
                if error_el:
                    text = await error_el.inner_text()
                    if text.strip():
                        return f"Error on page: {text.strip()}"
            except Exception:
                pass

        try:
            text_indicators = ["invalid code", "expired", "invalid booking", "not found"]
            for indicator in text_indicators:
                element = await self._page.query_selector(f":has-text('{indicator}')")
                if element:
                    return f"Page indicates: {indicator}"
        except Exception:
            pass

        return ""

    async def _scrape_matches_from_page(self) -> List[Dict]:
        """Scrape all matches from the current page"""
        matches = []

        for selector in self.site_config.match_selectors:
            try:
                items = await self._page.query_selector_all(selector)
                if items:
                    logger.debug(f"Found {len(items)} matches with selector: {selector}")
                    break
            except Exception:
                continue
        else:
            logger.warning("No match items found with any selector")
            return matches

        for i, item in enumerate(items):
            try:
                match_data = await self._extract_match_details(item)
                if match_data and self._validate_match_data(match_data):
                    match_data["source_site"] = self.site_config.name
                    matches.append(match_data)
            except Exception as e:
                logger.error(f"Failed to parse match {i}: {e}")
                continue

        return matches

    async def _extract_match_details(self, item) -> Dict:
        """Extract details from a single match item"""
        home_team = await self._get_text_from_selectors(item, self.site_config.team_name_selectors["home"])
        away_team = await self._get_text_from_selectors(item, self.site_config.team_name_selectors["away"])

        odds_home = await self._get_numeric_from_selectors(item, self.site_config.odds_selectors["home"])
        odds_draw = await self._get_numeric_from_selectors(item, self.site_config.odds_selectors["draw"])
        odds_away = await self._get_numeric_from_selectors(item, self.site_config.odds_selectors["away"])

        if odds_home == 0.0 and odds_draw == 0.0 and odds_away == 0.0:
            odds_home, odds_draw, odds_away = await self._extract_odds_fallback(item)

        start_time = await self._get_element_text(item, "[data-start-time], .match-time, .start-time")
        if not start_time:
            start_time = datetime.now(timezone.utc).isoformat()

        home_team = " ".join(home_team.split())
        away_team = " ".join(away_team.split())

        return {
            "home_team": home_team,
            "away_team": away_team,
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "start_time": start_time,
            "status": "upcoming"
        }

    async def _get_text_from_selectors(self, item, selectors: List[str]) -> str:
        """Get text from element using multiple selectors"""
        for selector in selectors:
            try:
                el = await item.query_selector(selector)
                if el:
                    text = await el.inner_text()
                    if text.strip():
                        return text.strip()
            except Exception:
                continue
        return "Unknown"

    async def _get_numeric_from_selectors(self, item, selectors: List[str]) -> float:
        """Get numeric value from element using multiple selectors"""
        for selector in selectors:
            try:
                el = await item.query_selector(selector)
                if el:
                    text = await el.inner_text()
                    text = text.strip().replace(",", ".")
                    return float(text) if text else 0.0
            except (ValueError, TypeError):
                continue
        return 0.0

    async def _get_element_text(self, item, selector: str) -> str:
        """Get text from element using a single selector"""
        try:
            el = await item.query_selector(selector)
            if el:
                return await el.inner_text()
        except Exception:
            pass
        return ""

    async def _extract_odds_fallback(self, item) -> Tuple[float, float, float]:
        """Fallback odds extraction using text parsing"""
        odds = [0.0, 0.0, 0.0]

        try:
            odds_elements = await item.query_selector_all(".m-odds-num, .odds, .odds-number, [data-testid='odds']")
            if len(odds_elements) >= 3:
                texts = []
                for el in odds_elements[:3]:
                    text = await el.inner_text()
                    text = text.strip().replace(",", ".")
                    try:
                        texts.append(float(text))
                    except ValueError:
                        texts.append(0.0)
                odds = tuple(texts)
        except Exception:
            pass

        return tuple(odds)

    def _validate_match_data(self, match: Dict) -> bool:
        """Validate match data has required fields"""
        if not isinstance(match, dict):
            return False

        required_fields = ["home_team", "away_team"]
        for field_name in required_fields:
            if field_name not in match or not match[field_name]:
                return False

        if "Unknown" in str(match.get("home_team", "")):
            return False
        if "Unknown" in str(match.get("away_team", "")):
            return False

        return True

    async def wait_random_delay(self) -> None:
        """Wait for a random delay between requests"""
        delay = random.uniform(self.config.min_delay, self.config.max_delay)
        await asyncio.sleep(delay)


async def extract_all_sites(
    urls: List[str],
    bookmaker: Bookmaker = Bookmaker.CUSTOM,
    config: Optional[ScraperConfig] = None,
    output_directory: Optional[str] = None,
    parallel: bool = True,
    max_parallel: int = 3
) -> List[ScrapeResult]:
    """
    Extract data from multiple sites in parallel

    Args:
        urls: List of URLs with optional betting codes
        bookmaker: Target bookmaker type
        config: Scraper configuration
        output_directory: Directory to save JSON outputs
        parallel: Whether to scrape in parallel
        max_parallel: Maximum concurrent scrapes

    Returns:
        List of ScrapeResult objects
    """
    if not config:
        config = ScraperConfig()

    site_config = SITE_CONFIGS.get(bookmaker, SITE_CONFIGS[Bookmaker.CUSTOM])
    site_config.name = site_config.name or bookmaker.value

    semaphore = asyncio.Semaphore(max_parallel)

    async def scrape_single(url: str) -> ScrapeResult:
        """Scrape a single URL with semaphore control"""
        async with semaphore:
            betting_code = None
            if "=" in url:
                betting_code = url.split("=")[-1].split("&")[0]

            async with MultiSiteScraper(config=config, site_config=site_config) as scraper:
                return await scraper.extract_from_url(url, betting_code)

    if parallel:
        tasks = [scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ScrapeResult(
                    success=False,
                    matches=[],
                    source_url=urls[i],
                    source_site=site_config.name,
                    error_message=str(result),
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))
            else:
                processed_results.append(result)
        results = processed_results
    else:
        results = []
        for url in urls:
            try:
                result = await scrape_single(url)
                results.append(result)
            except Exception as e:
                results.append(ScrapeResult(
                    success=False,
                    matches=[],
                    source_url=url,
                    source_site=site_config.name,
                    error_message=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat()
                ))

    if output_directory:
        await save_results_to_json(results, output_directory)

    return results


async def save_results_to_json(results: List[ScrapeResult], output_directory: str) -> List[str]:
    """
    Save scrape results to JSON files

    Args:
        results: List of ScrapeResult objects
        output_directory: Directory to save files

    Returns:
        List of saved file paths
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for i, result in enumerate(results):
        timestamp = result.timestamp.replace(":", "-").replace(".", "-")
        filename = f"scrape_{timestamp}_{i}.json"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            f.write(result.to_json(include_raw_html=False))

        saved_files.append(str(filepath))

    # Save consolidated results
    consolidated = {
        "scraping_session": {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scraper_version": "2.0.0",
            "total_sites": len(results),
            "successful_scrapes": sum(1 for r in results if r.success),
            "failed_scrapes": sum(1 for r in results if not r.success)
        },
        "results": [r.to_dict() for r in results]
    }

    consolidated_file = output_path / "consolidated_results.json"
    with open(consolidated_file, "w") as f:
        json.dump(consolidated, f, indent=2)

    saved_files.append(str(consolidated_file))

    logger.info(f"Saved {len(saved_files)} files to {output_directory}")

    return saved_files


def log_scraping_result(matches: List[Dict], success: bool, message: str = "") -> None:
    """Log scraping results consistently"""
    status = "SUCCESS" if success else "FAILED"
    count = len(matches)
    logger.info(f"[{status}] Scraped {count} matches. {message}")


def extract_from_url_sync(
    url: str,
    betting_code: Optional[str] = None,
    config: Optional[ScraperConfig] = None,
    bookmaker: Bookmaker = Bookmaker.CUSTOM
) -> ScrapeResult:
    """
    Synchronous wrapper for extract_from_url

    Args:
        url: Target URL
        betting_code: Optional betting code
        config: Scraper configuration
        bookmaker: Target bookmaker

    Returns:
        ScrapeResult object
    """
    if not config:
        config = ScraperConfig()

    site_config = SITE_CONFIGS.get(bookmaker, SITE_CONFIGS[Bookmaker.CUSTOM])

    async def _run():
        async with MultiSiteScraper(config=config, site_config=site_config) as scraper:
            return await scraper.extract_from_url(url, betting_code)

    return asyncio.run(_run())


async def extract_sportybet_ng(code: str, config: Optional[ScraperConfig] = None) -> ScrapeResult:
    """
    Extract data from SportyBet Nigeria using a betting code

    Args:
        code: SportyBet booking code
        config: Optional custom configuration

    Returns:
        ScrapeResult with extracted matches
    """
    if not config:
        config = ScraperConfig(headless=True, output_include_raw_html=False)

    site_config = SITE_CONFIGS[Bookmaker.SPORTYBET_NG]
    url = f"{site_config.base_url}/play/{code}"

    async with MultiSiteScraper(config=config, site_config=site_config) as scraper:
        return await scraper.extract_from_url(url, code)


async def extract_bet9ja(code: str, config: Optional[ScraperConfig] = None) -> ScrapeResult:
    """
    Extract data from Bet9ja using a betting code

    Args:
        code: Bet9ja betting code
        config: Optional custom configuration

    Returns:
        ScrapeResult with extracted matches
    """
    if not config:
        config = ScraperConfig(headless=True)

    site_config = SITE_CONFIGS[Bookmaker.BET9JA]
    url = f"{site_config.base_url}?{site_config.code_param}={code}"

    async with MultiSiteScraper(config=config, site_config=site_config) as scraper:
        return await scraper.extract_from_url(url, code)


async def extract_nairabet(code: str, config: Optional[ScraperConfig] = None) -> ScrapeResult:
    """
    Extract data from Nairabet using a betting code

    Args:
        code: Nairabet betting code
        config: Optional custom configuration

    Returns:
        ScrapeResult with extracted matches
    """
    if not config:
        config = ScraperConfig(headless=True)

    site_config = SITE_CONFIGS[Bookmaker.NAIRABET]
    url = f"{site_config.base_url}/play/{code}"

    async with MultiSiteScraper(config=config, site_config=site_config) as scraper:
        return await scraper.extract_from_url(url, code)


# Legacy function name for backward compatibility
async def legacy_extract_from_all_sites(code: str, config: Optional[ScraperConfig] = None) -> List[Dict[str, Any]]:
    """
    Legacy function - use extract_all_sites() for new code.
    Extracts data from all configured sites in parallel.

    Args:
        code: SportyBet booking code
        config: Scraper configuration

    Returns:
        List of results from each site
    """
    config = config or ScraperConfig()
    urls = [tmpl.format(code=code) for tmpl in config.sites.values()]
    results = await extract_all_sites(urls, config=config)
    return [r.to_dict() for r in results]


# Legacy function for backward compatibility
async def legacy_extract_sporty_code(code: str) -> List[Dict]:
    """
    Legacy main entry point for SportyBet code extraction.
    Uses primary site (SportyBet Nigeria)

    Args:
        code: SportyBet booking code

    Returns:
        List of matches with odds
    """
    config = ScraperConfig()
    primary_site = "sportybet_ng"
    url_template = config.sites.get(primary_site, "https://www.sportybet.com/ng/play/{code}")

    async with MultiSiteScraper(config) as scraper:
        result = await scraper.extract_from_url(url_template.format(code=code), code)

        if result.error_message:
            logger.warning(f"Primary site error: {result.error_message}")

        return result.matches if result.success else []


__all__ = [
    # Core classes
    "ScraperConfig",
    "ScraperSiteConfig",
    "Bookmaker",
    "ScrapeResult",
    "MultiSiteScraper",

    # Core functions
    "extract_all_sites",
    "save_results_to_json",
    "extract_from_url_sync",

    # Site-specific convenience functions
    "extract_sportybet_ng",
    "extract_bet9ja",
    "extract_nairabet",

    # Constants and utilities
    "SITE_CONFIGS",
    "ScrapingError",
    "log_scraping_result",

    # Legacy exports for backward compatibility (renamed to avoid conflict)
    "legacy_extract_from_all_sites",
    "legacy_extract_sporty_code"
]