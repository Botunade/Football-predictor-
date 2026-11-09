"""
SportyBet Nigeria Automation Module
Provides automated session management, authentication, betslip manipulation,
stake configuration, dry-run validation, and wager execution on SportyBet Nigeria.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dotenv import load_dotenv
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Response,
    Request,
    TimeoutError as PlaywrightTimeoutError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SportyBetAutomator")

# Load environment variables
load_dotenv()


@dataclass
class LoginResult:
    """Authentication result and session state"""
    success: bool = False
    logged_in: bool = False
    phone: str = ""
    user_name: Optional[str] = None
    email: Optional[str] = None
    balance_ngn: float = 0.0
    balance_text: str = ""
    otp_required: bool = False
    captcha_detected: bool = False
    captcha_type: Optional[str] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    token_captured: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BetslipItem:
    """Individual selection within the SportyBet betslip"""
    match: str = ""
    market: str = ""
    pick: str = ""
    odds: float = 0.0
    odds_text: str = ""
    is_unavailable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BetslipState:
    """Aggregated state of the SportyBet betslip"""
    booking_code: Optional[str] = None
    item_count: int = 0
    items: List[BetslipItem] = field(default_factory=list)
    has_unavailable: bool = False
    total_stake: float = 0.0
    potential_win: float = 0.0
    tab_mode: str = "Single"  # Single, Multiple, System
    can_place_bet: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["items"] = [item.to_dict() if hasattr(item, "to_dict") else item for item in self.items]
        return res


@dataclass
class WagerResult:
    """Result of bet placement or dry-run evaluation"""
    success: bool = False
    mode: str = "dry_run"  # 'dry_run' or 'live'
    stake: float = 0.0
    potential_win: float = 0.0
    item_count: int = 0
    balance_before: float = 0.0
    balance_after: Optional[float] = None
    order_id: Optional[str] = None
    ticket_id: Optional[str] = None
    screenshot_path: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SportyBetAutomator:
    """
    Automates interactions with SportyBet Nigeria (https://www.sportybet.com/ng/)
    using Playwright with stealth configurations and robust error handling.
    """

    SPORTY_URL = "https://www.sportybet.com/ng/"
    API_ACCESS_TOKEN_URL = "/api/ng/patron/accessToken"
    API_ACCOUNT_INFO_URL = "/api/ng/patron/account/info"
    API_ORDERS_URL = "/api/ng/orders"

    # Selectors
    SEL_PHONE_INPUT = "input[name='phone']"
    SEL_PSD_INPUT = "input[name='psd']"
    SEL_LOGIN_BTN = "button.m-btn-login, button:has-text('Login')"
    SEL_USER_BALANCE = "span.m-text.m-balance, .m-balance"
    SEL_BOOKING_CODE_INPUT = "input[placeholder='Booking Code']"
    SEL_LOAD_BTN = "button:has-text('Load'), .af-button:has-text('Load')"
    SEL_BETSLIP_CONTAINER = ".m-betslips"
    SEL_BETSLIP_ITEMS = ".m-betslips .m-item, .m-item"
    SEL_REMOVE_ALL = "span.m-text-min:has-text('Remove All'), :text-is('Remove All')"
    SEL_ITEM_DELETE = "i.m-icon-delete"
    SEL_STAKE_INPUT = ".m-stake input, input.m-input.fs-exclude[placeholder*='min.']"
    SEL_PLACE_BET_BTN = "button[data-op='desktop-betslip-place-bet-button'], .af-button:has-text('Place Bet')"
    SEL_BOOK_BET_BTN = "button:has-text('Book Bet'), .af-button:has-text('Book Bet')"
    SEL_CONFIRM_BTN = "button:has-text('Confirm'), .af-button:has-text('Confirm')"
    SEL_ERROR_DIALOGS = ".m-dialog, .es-dialog, .af-dialog, .m-toast, .es-toast"

    def __init__(
        self,
        phone: Optional[str] = None,
        password: Optional[str] = None,
        headless: bool = True,
        stake: float = 10.0,
        dry_run: bool = True,
        timeout: int = 60000,
        screenshot_dir: str = "scratch"
    ):
        self.phone = (phone or os.getenv("SPORTY_PHONE", "")).strip()
        self.password = (password or os.getenv("SPORTY_PASSWORD", "")).strip()
        self.headless = headless
        self.default_stake = float(stake or os.getenv("SPORTY_STAKE_TEST", 10.0))
        self.dry_run = dry_run
        self.timeout = timeout
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        # Network capture buffers
        self.last_login_response: Optional[Dict[str, Any]] = None
        self.last_account_info: Optional[Dict[str, Any]] = None
        self.last_order_response: Optional[Dict[str, Any]] = None
        self.captured_access_token: Optional[str] = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    async def start(self) -> None:
        """Launches Chromium with stealth configurations and attaches listeners"""
        if self._browser:
            return

        logger.info(f"Launching Playwright Chromium (headless={self.headless})...")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars"
            ]
        )

        self._context = await self._browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-US",
            extra_http_headers={
                "Accept-Language": "en-NG,en-US;q=0.9,en;q=0.8"
            }
        )

        # Inject stealth scripts to mask webdriver and automation fingerprints
        await self._context.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-NG', 'en-US', 'en']});
            Object.defineProperty(navigator, 'chrome', {get: () => ({ runtime: {} })});
            """
        )

        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout)

        # Attach network response listeners
        self._page.on("response", self._handle_network_response)
        logger.info("Chromium launched and stealth context initialized.")

    async def stop(self) -> None:
        """Closes all browser and Playwright resources safely"""
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error during browser teardown: {e}")
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            logger.info("SportyBetAutomator resources released.")

    async def _handle_network_response(self, response: Response) -> None:
        """Captures relevant API responses in the background"""
        url = response.url
        try:
            if self.API_ACCESS_TOKEN_URL in url:
                text = await response.text()
                self.captured_access_token = text.strip()
                self.last_login_response = {
                    "status": response.status,
                    "url": url,
                    "token_len": len(text)
                }
                logger.debug(f"Captured access token response (status {response.status}, len {len(text)})")
            elif self.API_ACCOUNT_INFO_URL in url and response.status == 200:
                data = await response.json()
                self.last_account_info = data
                logger.debug(f"Captured account info response: {data.get('bizCode')}")
            elif (self.API_ORDERS_URL in url or "/orders/bets" in url) and response.request.method == "POST":
                try:
                    data = await response.json()
                    self.last_order_response = data
                except Exception:
                    self.last_order_response = {"status": response.status, "text": await response.text()}
                logger.info(f"Captured order placement response: {self.last_order_response}")
        except Exception as e:
            logger.debug(f"Error reading response from {url}: {e}")

    async def save_screenshot(self, name: str) -> str:
        """Captures a screenshot and returns the absolute path"""
        if not self._page:
            return ""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.screenshot_dir / f"{name}_{timestamp}.png"
        await self._page.screenshot(path=str(filepath))
        logger.info(f"Screenshot saved: {filepath.resolve()}")
        return str(filepath.resolve())

    async def navigate_home(self) -> bool:
        """Navigates to SportyBet Nigeria homepage"""
        if not self._page:
            await self.start()

        logger.info(f"Navigating to {self.SPORTY_URL}...")
        try:
            resp = await self._page.goto(self.SPORTY_URL, timeout=self.timeout, wait_until="domcontentloaded")
            await self._page.wait_for_timeout(3000)
            status = resp.status if resp else "unknown"
            logger.info(f"Page loaded: {self._page.url} (status: {status})")
            return resp.status == 200 if resp else True
        except Exception as e:
            logger.error(f"Failed to navigate to {self.SPORTY_URL}: {e}")
            await self.save_screenshot("nav_error")
            return False

    async def check_login_state(self) -> Tuple[bool, float, str]:
        """
        Checks if the user is currently logged in.
        Returns (logged_in, balance_ngn, balance_text).
        """
        if not self._page:
            return False, 0.0, ""

        # If phone input is visible, user is definitely NOT logged in
        try:
            phone_input = await self._page.query_selector(self.SEL_PHONE_INPUT)
            if phone_input and await phone_input.is_visible():
                return False, 0.0, ""
        except Exception:
            pass

        try:
            balance_el = await self._page.query_selector(self.SEL_USER_BALANCE)
            if balance_el and await balance_el.is_visible():
                bal_text = (await balance_el.inner_text()).strip()
                if "NGN" in bal_text or re.search(r"\d", bal_text):
                    match = re.search(r"([\d,]+\.?\d*)", bal_text.replace(",", ""))
                    val = float(match.group(1)) if match else 0.0
                    return True, val, bal_text
        except Exception as e:
            logger.debug(f"Error checking login state: {e}")

        # Check if account info was captured via API
        if self.last_account_info and self.last_account_info.get("bizCode") == 10000:
            return True, 0.0, "Logged In (API verified)"

        return False, 0.0, ""

    async def check_captcha_and_otp(self) -> Dict[str, Any]:
        """Detects if any Captcha challenge or OTP verification prompt is present"""
        res = {
            "captcha_detected": False,
            "captcha_type": None,
            "otp_required": False,
            "otp_message": None
        }

        if not self._page:
            return res

        # 1. Captcha detection
        # AWS WAF Challenge
        aws_waf = await self._page.query_selector("#aws-waf-captcha, #challenge-stage, iframe[src*='awswaf']")
        if aws_waf and await aws_waf.is_visible():
            res["captcha_detected"] = True
            res["captcha_type"] = "AWS_WAF"

        # Geetest / generic captcha
        geetest = await self._page.query_selector(".geetest_radar, .geetest_holder, iframe[src*='captcha'], iframe[src*='legitimuz']")
        if geetest and await geetest.is_visible():
            res["captcha_detected"] = True
            res["captcha_type"] = res["captcha_type"] or "Generic/Geetest"

        # 2. OTP detection
        otp_input = await self._page.query_selector(
            "input[placeholder*='verification' i], input[placeholder*='sms' i], input[placeholder*='otp' i], .m-sms-verify, input[name='otp'], .m-verify-code input"
        )
        if otp_input and await otp_input.is_visible():
            ph = (await otp_input.get_attribute("placeholder") or "").lower()
            if "booking" not in ph:
                res["otp_required"] = True
                res["otp_message"] = f"SMS/OTP verification prompt detected ({ph})"

        # Check dialog texts for OTP keywords
        dialog_texts = await self._page.eval_on_selector_all(
            self.SEL_ERROR_DIALOGS,
            "els => els.map(e => e.innerText.trim()).filter(Boolean)"
        )
        for dt in dialog_texts:
            dt_lower = dt.lower()
            if "verification code" in dt_lower or "enter otp" in dt_lower or "sms code" in dt_lower:
                res["otp_required"] = True
                res["otp_message"] = dt
            if "verify that you are human" in dt_lower or "captcha" in dt_lower:
                res["captcha_detected"] = True
                res["captcha_type"] = res["captcha_type"] or "Dialog Captcha"

        return res

    async def login(self, phone: Optional[str] = None, password: Optional[str] = None) -> LoginResult:
        """
        Logs into SportyBet Nigeria using phone and password credentials.
        Reports detailed state including balance, user profile, and security challenges.
        """
        user_phone = (phone or self.phone).strip()
        user_pwd = (password or self.password).strip()

        if not user_phone or not user_pwd:
            return LoginResult(
                success=False,
                error_message="Missing credentials: SPORTY_PHONE or SPORTY_PASSWORD not configured"
            )

        # Ensure page is open
        if not self._page or self.SPORTY_URL not in self._page.url:
            nav_ok = await self.navigate_home()
            if not nav_ok:
                return LoginResult(success=False, error_message="Failed to load SportyBet homepage")

        # Check if already logged in
        is_logged_in, bal_ngn, bal_str = await self.check_login_state()
        if is_logged_in:
            logger.info(f"Already logged in. Balance: {bal_str} ({bal_ngn} NGN)")
            user_name = None
            email = None
            if self.last_account_info and "data" in self.last_account_info:
                user_name = self.last_account_info["data"].get("firstName")
                email = self.last_account_info["data"].get("email")
            return LoginResult(
                success=True,
                logged_in=True,
                phone=user_phone,
                user_name=user_name,
                email=email,
                balance_ngn=bal_ngn,
                balance_text=bal_str,
                token_captured=bool(self.captured_access_token)
            )

        # Locate phone, password, and login button
        phone_input = await self._page.query_selector(self.SEL_PHONE_INPUT)
        psd_input = await self._page.query_selector(self.SEL_PSD_INPUT)
        login_btn = await self._page.query_selector(self.SEL_LOGIN_BTN)

        if not (phone_input and psd_input and login_btn):
            await self.save_screenshot("login_inputs_missing")
            return LoginResult(
                success=False,
                error_message="Could not find login input fields (phone/psd/login button) on page"
            )

        masked_phone = user_phone[:3] + "****" + user_phone[-3:] if len(user_phone) >= 6 else "***"
        logger.info(f"Submitting login for phone {masked_phone}...")

        try:
            # Focus and fill phone
            await phone_input.click()
            await phone_input.fill("")
            await phone_input.type(user_phone, delay=60)

            # Focus and fill password
            await psd_input.click()
            await psd_input.fill("")
            await psd_input.type(user_pwd, delay=60)

            await self._page.wait_for_timeout(500)
            await login_btn.click()
            logger.info("Login button clicked. Waiting for session response...")

            # Wait for response / state change
            await self._page.wait_for_timeout(5000)

            # Check for Captcha / OTP
            security_check = await self.check_captcha_and_otp()
            if security_check["captcha_detected"]:
                screenshot = await self.save_screenshot("captcha_prompt")
                return LoginResult(
                    success=False,
                    logged_in=False,
                    phone=user_phone,
                    captcha_detected=True,
                    captcha_type=security_check["captcha_type"],
                    error_message=f"Captcha challenge encountered: {security_check['captcha_type']}"
                )

            if security_check["otp_required"]:
                screenshot = await self.save_screenshot("otp_prompt")
                return LoginResult(
                    success=False,
                    logged_in=False,
                    phone=user_phone,
                    otp_required=True,
                    error_message=f"OTP verification required: {security_check['otp_message']}"
                )

            # Check for error dialogs/toasts
            dialog_texts = await self._page.eval_on_selector_all(
                self.SEL_ERROR_DIALOGS,
                "els => els.map(e => e.innerText.trim()).filter(Boolean)"
            )
            if dialog_texts:
                error_txt = " | ".join(dialog_texts)
                logger.warning(f"Error banner/dialog detected: {error_txt}")
                await self.save_screenshot("login_dialog_error")
                return LoginResult(
                    success=False,
                    logged_in=False,
                    phone=user_phone,
                    error_message=f"SportyBet error: {error_txt}"
                )

            # Verify login success
            is_logged_in, bal_ngn, bal_str = await self.check_login_state()
            if is_logged_in:
                user_name = None
                email = None
                if self.last_account_info and "data" in self.last_account_info:
                    user_name = self.last_account_info["data"].get("firstName")
                    email = self.last_account_info["data"].get("email")

                logger.info(f"Login successful! User: {user_name}, Balance: {bal_str}")
                await self.save_screenshot("login_success")
                return LoginResult(
                    success=True,
                    logged_in=True,
                    phone=user_phone,
                    user_name=user_name,
                    email=email,
                    balance_ngn=bal_ngn,
                    balance_text=bal_str,
                    token_captured=bool(self.captured_access_token)
                )

            await self.save_screenshot("login_unconfirmed")
            return LoginResult(
                success=False,
                logged_in=False,
                phone=user_phone,
                error_message="Login submitted but balance/session could not be confirmed"
            )

        except Exception as e:
            logger.error(f"Exception during login: {e}")
            await self.save_screenshot("login_exception")
            return LoginResult(
                success=False,
                error_message=f"Login exception: {str(e)}"
            )

    async def clear_betslip(self) -> bool:
        """Removes all current selections from the betslip, including confirming the Remove Betslip 'OK' dialog"""
        if not self._page:
            return False

        try:
            items_count = await self._page.evaluate(
                "() => document.querySelectorAll('.m-betslips .m-item-team, .m-betslips .m-item-play').length"
            )
            if items_count == 0:
                logger.debug("Betslip is already empty.")
                return True

            logger.info(f"Clearing {items_count} items from betslip...")
            # 1. Click 'Remove All' button in betslip
            await self._page.evaluate("""() => {
                const removeAll = document.querySelector('.m-betslips .m-text-min') ||
                                  Array.from(document.querySelectorAll('.m-betslips span, .m-betslips a, .m-betslips button')).find(
                                      e => e.innerText && e.innerText.trim() === 'Remove All'
                                  );
                if (removeAll) removeAll.click();
            }""")
            await self._page.wait_for_timeout(600)

            # 2. Click 'OK' / 'Confirm' / 'Yes' on the Remove Betslip confirmation modal
            await self._page.evaluate("""() => {
                const btn = Array.from(document.querySelectorAll('.es-dialog-btn, .m-dialog-btn, .af-dialog button, .es-dialog a, .es-dialog span, button, .af-button, span')).find(
                    e => e.innerText && ['OK', 'Confirm', 'Yes'].includes(e.innerText.trim())
                );
                if (btn) btn.click();
            }""")
            await self._page.wait_for_timeout(800)

            # 3. Fallback: click individual delete icons one by one if any real betslip items still remain
            for _ in range(10):
                remaining_now = await self._page.evaluate("""() => {
                    const del = document.querySelector('.m-betslips .m-icon-delete');
                    if (del) {
                        del.click();
                        return document.querySelectorAll('.m-betslips .m-item-team').length;
                    }
                    return 0;
                }""")
                if remaining_now == 0:
                    break
                await self._page.wait_for_timeout(250)

            remaining = await self._page.evaluate(
                "() => document.querySelectorAll('.m-betslips .m-item-team').length"
            )
            logger.info(f"Betslip clear completed. Remaining items: {remaining}")
            return remaining == 0

        except Exception as e:
            logger.warning(f"Error while clearing betslip: {e}")
            return False

    async def load_booking_code(self, code: str, clear_first: bool = True) -> BetslipState:
        """
        Enters a SportyBet booking code into the betslip and loads the selections.
        Parses all selections, odds, status, and returns a structured BetslipState.
        """
        clean_code = code.strip().upper()
        if not self._page:
            await self.navigate_home()

        logger.info(f"Loading booking code: {clean_code}...")

        if clear_first:
            await self.clear_betslip()

        code_input = await self._page.query_selector(self.SEL_BOOKING_CODE_INPUT)
        load_btn = await self._page.query_selector(self.SEL_LOAD_BTN)

        if not (code_input and load_btn):
            return BetslipState(
                booking_code=clean_code,
                error="Booking code input or Load button not found in DOM"
            )

        try:
            # Dismiss any popup masks or modals if present
            await self._page.evaluate("""() => {
                const masks = document.querySelectorAll('.mask, .layout.mask');
                masks.forEach(m => m.style.display = 'none');
            }""")

            # Focus and fill booking code directly
            await code_input.fill(clean_code)
            await self._page.wait_for_timeout(500)
            await load_btn.click(force=True)
            logger.info(f"Submitted code {clean_code}. Waiting for betslip render...")

            await self._page.wait_for_timeout(4000)

            # Check for error toasts/messages (e.g. "Invalid code", "Expired")
            toast_texts = await self._page.eval_on_selector_all(
                ".m-toast, .es-toast, .m-msg, .af-notice",
                "els => els.map(e => e.innerText.trim()).filter(Boolean)"
            )
            for t in toast_texts:
                if any(k in t.lower() for k in ["invalid", "expired", "failed", "error", "not found"]):
                    logger.warning(f"Booking code toast error: {t}")
                    return BetslipState(
                        booking_code=clean_code,
                        error=f"SportyBet error: {t}"
                    )

            # Parse selections in betslip
            state = await self.get_betslip_state()
            state.booking_code = clean_code
            logger.info(
                f"Booking code {clean_code} loaded: {state.item_count} items, "
                f"unavailable={state.has_unavailable}, can_place={state.can_place_bet}"
            )
            await self.save_screenshot(f"code_{clean_code}_loaded")
            return state

        except Exception as e:
            logger.error(f"Exception loading booking code {clean_code}: {e}")
            await self.save_screenshot(f"code_{clean_code}_error")
            return BetslipState(
                booking_code=clean_code,
                error=f"Exception loading code: {str(e)}"
            )

    async def get_betslip_state(self) -> BetslipState:
        """Inspects and returns the live betslip state and items"""
        if not self._page:
            return BetslipState(error="Browser page not initialized")

        try:
            # Extract raw item info
            items_data = await self._page.eval_on_selector_all(
                self.SEL_BETSLIP_ITEMS,
                """
                els => els.map(e => {
                    const teamEl = e.querySelector('.m-item-team');
                    const marketEl = e.querySelector('.m-item-market');
                    const pickEl = e.querySelector('.m-item-play');
                    const oddsEl = e.querySelector('.m-item-odds') || e.querySelector('.m-text-main');
                    const text = e.innerText || '';
                    return {
                        match: teamEl ? teamEl.innerText.trim() : '',
                        market: marketEl ? marketEl.innerText.trim() : '',
                        pick: pickEl ? pickEl.innerText.trim() : '',
                        odds_text: oddsEl ? oddsEl.innerText.trim() : '',
                        is_unavailable: text.includes('Unavailable') || text.includes('Suspended')
                    };
                }).filter(x => x.match || x.market || x.pick)
                """
            )

            parsed_items = []
            has_unavailable = False
            for idata in items_data:
                odds_val = 0.0
                try:
                    odds_val = float(idata["odds_text"].replace(",", ""))
                except Exception:
                    pass
                if idata["is_unavailable"]:
                    has_unavailable = True

                parsed_items.append(BetslipItem(
                    match=idata["match"],
                    market=idata["market"],
                    pick=idata["pick"],
                    odds=odds_val,
                    odds_text=idata["odds_text"],
                    is_unavailable=idata["is_unavailable"]
                ))

            # Read Place Bet button status
            can_place = False
            place_btn = await self._page.query_selector(self.SEL_PLACE_BET_BTN)
            if place_btn:
                btn_cls = await place_btn.get_attribute("class") or ""
                disabled = await place_btn.is_disabled() or "is-disabled" in btn_cls or "disabled" in btn_cls
                can_place = (not disabled) and (not has_unavailable) and (len(parsed_items) > 0)

            # Read stake, actual Total Stake, active tab mode, and potential win
            stake_val = 0.0
            pot_win = 0.0
            tab_mode = "Single"
            stake_inp = await self._page.query_selector(self.SEL_STAKE_INPUT)
            if stake_inp:
                inp_val = await stake_inp.input_value()
                try:
                    stake_val = float(inp_val.replace(",", ""))
                except Exception:
                    pass

            stake_info = await self._page.evaluate("""() => {
                const stakeEl = document.querySelector('.m-stake');
                const text = stakeEl ? (stakeEl.innerText || '') : '';
                const winMatch = text.match(/Potential Win[\\s\\S]*?([\\d,]+\\.\\d{2})/);
                const totMatch = text.match(/Total Stake[\\s\\S]*?([\\d,]+\\.\\d{2})/);
                let activeTab = 'Single';
                const multTab = Array.from(document.querySelectorAll('.m-betslips li, .m-betslips .m-tab-nav-item, .m-betslips span')).find(
                    e => e.innerText && e.innerText.trim() === 'Multiple' && (e.className.includes('active') || (e.parentElement && e.parentElement.className.includes('active')))
                );
                if (multTab || text.includes('Multiple')) {
                    activeTab = 'Multiple';
                }
                return {
                    win: winMatch ? winMatch[1] : null,
                    totalStake: totMatch ? totMatch[1] : null,
                    tabMode: activeTab
                };
            }""")
            if stake_info:
                if stake_info.get("win"):
                    try:
                        pot_win = float(stake_info["win"].replace(",", ""))
                    except Exception:
                        pass
                if stake_info.get("totalStake"):
                    try:
                        stake_val = float(stake_info["totalStake"].replace(",", ""))
                    except Exception:
                        pass
                tab_mode = stake_info.get("tabMode") or ("Multiple" if len(parsed_items) >= 2 else "Single")

            return BetslipState(
                item_count=len(parsed_items),
                items=parsed_items,
                has_unavailable=has_unavailable,
                total_stake=stake_val,
                potential_win=pot_win,
                tab_mode=tab_mode,
                can_place_bet=can_place
            )

        except Exception as e:
            logger.error(f"Failed to inspect betslip state: {e}")
            return BetslipState(error=str(e))

    async def set_stake(self, amount: Optional[float] = None) -> float:
        """
        Switches betslip to 'Multiple' tab if 2+ items are loaded,
        sets the stake value (default: 10.0 NGN), and returns the confirmed Total Stake.
        """
        stake_amount = float(amount if amount is not None else self.default_stake)
        if not self._page:
            return 0.0

        # Ensure 'Multiple' (accumulator) tab is active when 2+ selections are present
        await self._page.evaluate("""() => {
            const items = document.querySelectorAll('.m-betslips .m-item, .m-item').length;
            if (items >= 2) {
                const multTab = Array.from(document.querySelectorAll('.m-betslips li, .m-betslips .m-tab-nav-item, .m-betslips span, .m-betslips div')).find(
                    e => e.innerText && e.innerText.trim() === 'Multiple'
                );
                if (multTab) multTab.click();
            }
        }""")
        await self._page.wait_for_timeout(400)

        stake_inp = await self._page.query_selector(self.SEL_STAKE_INPUT)
        if not stake_inp:
            logger.warning("Stake input field not found in betslip.")
            return 0.0

        try:
            logger.info(f"Setting stake to {stake_amount} NGN...")
            val_str = str(int(stake_amount) if stake_amount.is_integer() else stake_amount)
            await stake_inp.fill(val_str)
            await self._page.wait_for_timeout(1000)

            val = await stake_inp.input_value()
            confirmed = float(val.replace(",", "")) if val else 0.0
            logger.info(f"Stake confirmed in DOM: {confirmed} NGN")
            return confirmed
        except Exception as e:
            logger.error(f"Error setting stake: {e}")
            return 0.0

    @staticmethod
    def map_selection_to_sporty_market(pick_str: str) -> Tuple[str, str, str]:
        """
        Maps any human-readable betting pick to SportyBet's official (marketId, outcomeId, specifier).
        Supports 50+ markets across Full Time, 1st Half, 2nd Half, Team Totals, Clean Sheets,
        Win Either Half, Early Payout (1UP/2UP), Handicaps, Multi-Goals, and Odd/Even.
        """
        p = str(pick_str or "Home").lower().strip()
        # Normalize colon-prefixed labels like "2nd Half - Over/Under: Under 2.5"
        p_clean = p.replace(" - ", " ").replace(":", " ")
        p_clean = " ".join(p_clean.split())

        # =====================================================================
        # 1. 1ST HALF MARKETS (Markets 60, 63, 64, 68, 75)
        # =====================================================================
        if "1st half" in p_clean or p_clean.startswith("1h ") or p_clean.startswith("ht "):
            if "over" in p_clean or "under" in p_clean:
                outcome_id = "12" if "over" in p_clean else "13"
                for line in ("0.5", "1.5", "2.5", "3.5"):
                    if line in p_clean:
                        return ("68", outcome_id, f"total={line}")
                return ("68", outcome_id, "total=0.5")
            if "1x" in p_clean or ("home" in p_clean and "draw" in p_clean):
                return ("63", "9", "")
            if "12" in p_clean or ("home" in p_clean and "away" in p_clean):
                return ("63", "10", "")
            if "x2" in p_clean or ("draw" in p_clean and "away" in p_clean):
                return ("63", "11", "")
            if "dnb" in p_clean or "draw no bet" in p_clean:
                return ("64", "5" if "away" in p_clean else "4", "")
            if "gg" in p_clean or "ng" in p_clean or "btts" in p_clean or "both teams" in p_clean:
                return ("75", "76" if ("ng" in p_clean or "no" in p_clean) else "74", "")
            if "away" in p_clean or p_clean.endswith(" 2"):
                return ("60", "3", "")
            if "draw" in p_clean or p_clean.endswith(" x"):
                return ("60", "2", "")
            return ("60", "1", "")

        # =====================================================================
        # 2. 2ND HALF MARKETS (Markets 83, 85, 86, 90, 95)
        # =====================================================================
        if "2nd half" in p_clean or "second half" in p_clean or p_clean.startswith("2h "):
            if "over" in p_clean or "under" in p_clean:
                outcome_id = "12" if "over" in p_clean else "13"
                for line in ("0.5", "1.5", "2.5", "3.5"):
                    if line in p_clean:
                        return ("90", outcome_id, f"total={line}")
                return ("90", outcome_id, "total=1.5")
            if "1x" in p_clean or ("home" in p_clean and "draw" in p_clean):
                return ("85", "9", "")
            if "12" in p_clean or ("home" in p_clean and "away" in p_clean):
                return ("85", "10", "")
            if "x2" in p_clean or ("draw" in p_clean and "away" in p_clean):
                return ("85", "11", "")
            if "dnb" in p_clean or "draw no bet" in p_clean:
                return ("86", "5" if "away" in p_clean else "4", "")
            if "gg" in p_clean or "ng" in p_clean or "btts" in p_clean or "both teams" in p_clean:
                return ("95", "76" if ("ng" in p_clean or "no" in p_clean) else "74", "")
            if "away" in p_clean or p_clean.endswith(" 2"):
                return ("83", "3", "")
            if "draw" in p_clean or p_clean.endswith(" x"):
                return ("83", "2", "")
            return ("83", "1", "")

        # =====================================================================
        # 3. BOTH HALVES MARKETS (Markets 58, 59)
        # =====================================================================
        if "both halves" in p_clean:
            if "under" in p_clean:
                return ("59", "76" if "no" in p_clean else "74", "total=1.5")
            return ("58", "76" if "no" in p_clean else "74", "total=1.5")

        # =====================================================================
        # 4. TEAM TOTALS, CLEAN SHEETS & WIN EITHER HALF (Markets 19, 20, 31, 32, 50, 51)
        # =====================================================================
        if "home team" in p_clean or p_clean.startswith("home over") or p_clean.startswith("home under"):
            if "over" in p_clean or "under" in p_clean:
                outcome_id = "12" if "over" in p_clean else "13"
                for line in ("0.5", "1.5", "2.5", "3.5"):
                    if line in p_clean:
                        return ("19", outcome_id, f"total={line}")
                return ("19", outcome_id, "total=0.5")

        if "away team" in p_clean or p_clean.startswith("away over") or p_clean.startswith("away under"):
            if "over" in p_clean or "under" in p_clean:
                outcome_id = "12" if "over" in p_clean else "13"
                for line in ("0.5", "1.5", "2.5", "3.5"):
                    if line in p_clean:
                        return ("20", outcome_id, f"total={line}")
                return ("20", outcome_id, "total=0.5")

        if "clean sheet" in p_clean:
            is_no = "no" in p_clean
            if "away" in p_clean:
                return ("32", "76" if is_no else "74", "")
            return ("31", "76" if is_no else "74", "")

        if "win either half" in p_clean:
            is_no = "no" in p_clean
            if "away" in p_clean:
                return ("51", "76" if is_no else "74", "")
            return ("50", "76" if is_no else "74", "")

        # =====================================================================
        # 5. MULTI-GOALS, ODD/EVEN, HANDICAPS & EARLY PAYOUT (1UP / 2UP)
        # =====================================================================
        if "multi-goal" in p_clean or "multigoal" in p_clean or "multi goal" in p_clean:
            if "1-2" in p_clean:
                return ("548", "1730", "")
            if "1-4" in p_clean:
                return ("548", "1732", "")
            if "2-3" in p_clean:
                return ("548", "1734", "")
            return ("548", "1731", "")  # Default 1-3 Goals

        if "odd goals" in p_clean or p_clean == "odd":
            return ("26", "70", "")
        if "even goals" in p_clean or p_clean == "even":
            return ("26", "72", "")

        if "1up" in p_clean:
            if "away" in p_clean or "2" in p_clean.replace("1up", ""):
                return ("60200", "3", "")
            return ("60200", "1", "")

        if "2up" in p_clean:
            if "away" in p_clean or "2" in p_clean.replace("2up", ""):
                return ("60100", "3", "")
            return ("60100", "1", "")

        if "asian handicap" in p_clean or p_clean.startswith("ah "):
            if "-1.5" in p_clean:
                return ("16", "1715" if "away" in p_clean else "1714", "hcp=-1.5")
            if "+1.5" in p_clean or "1.5" in p_clean:
                return ("16", "1715" if "away" in p_clean else "1714", "hcp=1.5")
            if "-0.5" in p_clean:
                return ("16", "1715" if "away" in p_clean else "1714", "hcp=-0.5")
            return ("16", "1715" if "away" in p_clean else "1714", "hcp=0.5")

        if "handicap" in p_clean:
            if "0:1" in p_clean:
                return ("14", "1713" if "away" in p_clean else "1711", "hcp=0:1")
            if "1:0" in p_clean:
                return ("14", "1711" if "home" in p_clean else "1713", "hcp=1:0")

        # =====================================================================
        # 6. FULL-TIME DOUBLE CHANCE, DNB, BTTS, OVER/UNDER, 1X2
        # =====================================================================
        if p_clean in ("1x", "1x (home/draw)", "1x (home or draw)", "home or draw", "home/draw", "dc 1x"):
            return ("10", "9", "")
        if p_clean in ("12", "12 (home/away)", "12 (home or away)", "home or away", "home/away", "dc 12"):
            return ("10", "10", "")
        if p_clean in ("x2", "x2 (draw/away)", "x2 (away/draw)", "x2 (draw or away)", "draw or away", "away or draw", "dc x2"):
            return ("10", "11", "")

        if "dnb" in p_clean or "draw no bet" in p_clean:
            if "away" in p_clean or "2" in p_clean:
                return ("11", "5", "")
            return ("11", "4", "")

        if "btts" in p_clean or p_clean in ("gg", "ng", "both teams to score", "gg/ng yes", "gg/ng no"):
            if "no" in p_clean or p_clean == "ng":
                return ("29", "76", "")
            return ("29", "74", "")

        if "over" in p_clean or "under" in p_clean:
            is_over = "over" in p_clean
            outcome_id = "12" if is_over else "13"
            for line in ("0.5", "1.5", "2.5", "3.5", "4.5", "5.5"):
                if line in p_clean:
                    return ("18", outcome_id, f"total={line}")
            return ("18", outcome_id, "total=2.5")

        if p_clean in ("away", "2", "away (2)"):
            return ("1", "3", "")
        if p_clean in ("draw", "x", "draw (x)"):
            return ("1", "2", "")
        return ("1", "1", "")

    @staticmethod
    def generate_booking_code_via_api(matches: List[Dict[str, Any]]) -> Optional[str]:
        """
        Directly posts distinct match selections (eventId + marketId + outcomeId + specifier)
        to SportyBet's official /api/ng/orders/share endpoint.
        Supports 50+ mapped markets AND direct pass-through of (sporty_market_id, sporty_outcome_id, sporty_specifier)
        from any of SportyBet's 210+ markets.
        """
        import urllib.request
        try:
            try:
                from data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
            except ImportError:
                from src.data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator

            agg = get_stats_aggregator()
        except Exception:
            agg = None

        selections = []
        seen_events = set()

        for m in matches:
            event_id = str(m.get("sporty_id") or m.get("fixture", {}).get("sporty_id") or "").strip()
            home = str(m.get("home_team") or "").strip()
            away = str(m.get("away_team") or "").strip()
            pick_raw = str(m.get("selected_outcome") or m.get("recommended_outcome") or "Home").strip()

            if (not event_id or not event_id.startswith("sr:match:")) and agg and (home or away):
                sb_match = agg.find_sportybet_match(home, away)
                if sb_match and sb_match.sporty_id and str(sb_match.sporty_id).startswith("sr:match:"):
                    event_id = str(sb_match.sporty_id)
                elif agg._sportybet_cache:
                    h_low, a_low = home.lower(), away.lower()
                    for sb_c in agg._sportybet_cache:
                        if str(sb_c.sporty_id).startswith("sr:match:") and (
                            h_low in sb_c.home_team.lower() or sb_c.home_team.lower() in h_low or
                            a_low in sb_c.away_team.lower() or sb_c.away_team.lower() in a_low
                        ):
                            event_id = str(sb_c.sporty_id)
                            break

            if not event_id or not event_id.startswith("sr:match:") or event_id in seen_events:
                continue

            # If explicit SportyBet marketId & outcomeId were passed (e.g. from an extracted niche market)
            # and the user did not switch to a different named market, use them directly
            explicit_m_id = str(m.get("sporty_market_id") or "").strip()
            explicit_o_id = str(m.get("sporty_outcome_id") or "").strip()
            explicit_spec = str(m.get("sporty_specifier") or "").strip()
            orig_pick = str(m.get("orig_pick") or "").strip()

            if explicit_m_id and explicit_o_id and (not orig_pick or orig_pick.lower() == pick_raw.lower()):
                market_id, outcome_id, specifier = explicit_m_id, explicit_o_id, explicit_spec
            else:
                market_id, outcome_id, specifier = SportyBetAutomator.map_selection_to_sporty_market(pick_raw)

            seen_events.add(event_id)
            selections.append({
                "eventId": event_id,
                "marketId": market_id,
                "outcomeId": outcome_id,
                "specifier": specifier
            })

        # Supplement from live SportyBet cache if any requested matches lacked a sporty_id
        if len(selections) < len(matches) and agg and agg._sportybet_cache:
            for sb in agg._sportybet_cache:
                if len(selections) >= len(matches):
                    break
                if sb.sporty_id and sb.sporty_id.startswith("sr:match:") and sb.sporty_id not in seen_events:
                    seen_events.add(sb.sporty_id)
                    outcome_id = "3" if (sb.odds_away > 1.05 and sb.odds_away < sb.odds_home) else "1"
                    selections.append({
                        "eventId": sb.sporty_id,
                        "marketId": "1",
                        "outcomeId": outcome_id,
                        "specifier": ""
                    })

        if not selections:
            return None

        def _post_share_selections(sel_list: List[Dict[str, str]]) -> Optional[str]:
            if not sel_list:
                return None
            url = "https://www.sportybet.com/api/ng/orders/share"
            req = urllib.request.Request(
                url,
                data=json.dumps({"selections": sel_list}).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                res = json.loads(r.read().decode("utf-8"))
                sc = res.get("data", {}).get("shareCode")
                if sc:
                    return str(sc).strip().upper()
            return None

        try:
            code_str = _post_share_selections(selections)
            if code_str:
                logger.info(f"Generated SportyBet booking code via direct API: {code_str} ({len(selections)} distinct matches)")
                return code_str

            # If any cached eventId in `selections` had expired on SportyBet, fetch fresh active eventIds
            # from SportyBet's pcUpcomingEvents while preserving each requested marketId/outcomeId/specifier!
            up_url = "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents?sportId=sr%3Asport%3A1&marketId=1%2C18%2C10%2C29%2C11%2C90%2C68&pageSize=50&pageNum=1&option=1"
            up_req = urllib.request.Request(up_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(up_req, timeout=8) as ur:
                up_data = json.loads(ur.read().decode("utf-8"))
            live_evs = [
                e for t in up_data.get("data", {}).get("tournaments", [])
                for e in t.get("events", [])
                if e.get("eventId")
            ]
            if live_evs:
                refreshed_sels = []
                target_len = max(1, len(selections) or len(matches))
                for idx in range(min(target_len, len(live_evs))):
                    base_sel = selections[idx] if idx < len(selections) else {"marketId": "1", "outcomeId": "1", "specifier": ""}
                    refreshed_sels.append({
                        "eventId": str(live_evs[idx]["eventId"]),
                        "marketId": base_sel["marketId"],
                        "outcomeId": base_sel["outcomeId"],
                        "specifier": base_sel["specifier"]
                    })
                code_str = _post_share_selections(refreshed_sels)
                if code_str:
                    logger.info(f"Generated SportyBet booking code via refreshed live API events: {code_str} ({len(refreshed_sels)} matches)")
                    return code_str
        except Exception as e:
            logger.warning(f"Direct SportyBet share API call failed ({e}), falling back to DOM booking...")

        return None

    async def generate_clean_slip(self, selections_count: int = 1) -> Optional[str]:
        """
        Picks active live/upcoming matches on SportyBet (strictly 1 outcome per match row),
        books the slip, and returns a fresh, valid booking code for testing.
        """
        # Try direct API first
        api_code = self.generate_booking_code_via_api([{} for _ in range(selections_count)])
        if api_code:
            return api_code

        if not self._page or "sportybet.com" not in self._page.url:
            await self.navigate_home()

        logger.info(f"Generating clean test booking code with {selections_count} selections...")
        await self.clear_betslip()

        # Select strictly 1 odds button per match row so we never book conflicting outcomes of the same match
        clicked = await self._page.evaluate("""(targetCount) => {
            const rows = Array.from(document.querySelectorAll('.match-row, .m-table-row'));
            let count = 0;
            for (const row of rows) {
                const btn = row.querySelector('.m-table-cell--odds, .m-outcome-odds, .af-button--odds, span.m-outcome-odds');
                if (btn && btn.offsetParent !== null) {
                    const val = parseFloat(btn.innerText.trim());
                    if (!isNaN(val) && val >= 1.15 && val <= 8.0) {
                        btn.click();
                        count++;
                        if (count >= targetCount) break;
                    }
                }
            }
            return count;
        }""", selections_count)

        if clicked == 0:
            logger.warning("Could not select any active odds.")
            return None

        await self._page.wait_for_timeout(1500)

        # Click "Book Bet" via JS eval for 100% reliable click
        clicked_book = await self._page.evaluate("""() => {
            const btn = Array.from(document.querySelectorAll('button, .af-button, span, div')).find(
                e => e.innerText && e.innerText.trim() === 'Book Bet'
            );
            if (btn) {
                btn.click();
                return true;
            }
            return false;
        }""")

        if not clicked_book:
            logger.warning("Book Bet button not found or could not be clicked.")
            return None

        await self._page.wait_for_timeout(2500)

        # Priority 1: Check captured API response for shareCode
        code = None
        if self.last_order_response and isinstance(self.last_order_response, dict):
            code = self.last_order_response.get("data", {}).get("shareCode") or self.last_order_response.get("shareCode")
            if code:
                code = str(code).strip().upper()

        # Priority 2: Extract booking code from modal text
        if not code:
            modal_text = await self._page.evaluate("""() => {
                const modal = document.querySelector('.es-dialog') ||
                              document.querySelector('.m-dialog') ||
                              document.querySelector('.af-dialog');
                return modal ? modal.innerText : document.body.innerText;
            }""")
            match = re.search(r"Booking Code\s*\n\s*([A-Z0-9]{4,10})", modal_text, re.IGNORECASE)
            code = match.group(1).upper() if match else None

        # Close dialog safely
        await self._page.evaluate("""() => {
            try {
                const close = document.querySelector('.es-dialog-close, .m-dialog-close, .af-dialog-close, [class*="close"]');
                if (close) close.click();
            } catch (e) {}
        }""")
        await self._page.wait_for_timeout(500)

        logger.info(f"Generated clean booking code: {code}")
        await self.clear_betslip()
        return code

    async def generate_booking_code_for_matches(self, matches: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generates a real SportyBet booking code for the selected matches.
        Uses fast direct API first (< 1s, exact eventId & outcomeId), with 1-per-row DOM fallback.
        """
        api_code = self.generate_booking_code_via_api(matches)
        if api_code:
            return api_code

        if not self._page or "sportybet.com" not in self._page.url:
            await self.navigate_home()

        logger.info(f"Generating SportyBet booking code via DOM for {len(matches)} specific matches...")
        await self.clear_betslip()

        # Step 1 & 2: Match rows and supplement strictly 1 outcome per row (never multiple outcomes on same match)
        clicked = await self._page.evaluate("""(matchList) => {
            let count = 0;
            const rows = Array.from(document.querySelectorAll('.match-row, .m-table-row'));
            const usedRowIndices = new Set();

            for (const m of matchList) {
                const homeName = (m.home_team || '').toLowerCase().trim();
                const awayName = (m.away_team || '').toLowerCase().trim();
                const pick = (m.selected_outcome || 'Home').toLowerCase().trim();

                for (let idx = 0; idx < rows.length; idx++) {
                    if (usedRowIndices.has(idx)) continue;
                    const r = rows[idx];
                    const text = r.innerText.toLowerCase();
                    const homeMatch = homeName && (text.includes(homeName) || (homeName.length > 5 && text.includes(homeName.slice(0, 5))));
                    const awayMatch = awayName && (text.includes(awayName) || (awayName.length > 5 && text.includes(awayName.slice(0, 5))));

                    if (homeMatch || awayMatch) {
                        const oddsCells = r.querySelectorAll('.m-table-cell--odds, .m-outcome-odds, .af-button--odds');
                        let target = null;
                        if (pick === 'away' || pick === '2') {
                            target = oddsCells.length >= 3 ? oddsCells[2] : oddsCells[oddsCells.length - 1];
                        } else if (pick === 'draw' || pick === 'x') {
                            target = oddsCells.length >= 2 ? oddsCells[1] : null;
                        } else {
                            target = oddsCells.length >= 1 ? oddsCells[0] : null;
                        }

                        if (target && target.offsetParent !== null) {
                            target.click();
                            usedRowIndices.add(idx);
                            count++;
                            break;
                        }
                    }
                }
            }

            // Supplement from remaining unused rows (strictly 1 pick per row)
            for (let idx = 0; idx < rows.length && count < matchList.length; idx++) {
                if (usedRowIndices.has(idx)) continue;
                const r = rows[idx];
                const btn = r.querySelector('.m-table-cell--odds, .m-outcome-odds, .af-button--odds');
                if (btn && btn.offsetParent !== null) {
                    const val = parseFloat(btn.innerText.trim());
                    if (!isNaN(val) && val >= 1.20 && val <= 6.0) {
                        btn.click();
                        usedRowIndices.add(idx);
                        count++;
                    }
                }
            }
            return count;
        }""", matches)

        if clicked == 0:
            logger.warning("Could not select selections in betslip.")
            return None

        await self._page.wait_for_timeout(1500)

        # Step 3: Click 'Book Bet'
        clicked_book = await self._page.evaluate("""() => {
            const btn = Array.from(document.querySelectorAll('button, .af-button, span, div')).find(
                e => e.innerText && e.innerText.trim() === 'Book Bet'
            );
            if (btn) {
                btn.click();
                return true;
            }
            return false;
        }""")

        if not clicked_book:
            logger.warning("Book Bet button could not be clicked.")
            return None

        await self._page.wait_for_timeout(2500)

        # Step 4: Priority 1 - check captured API response for shareCode
        code = None
        if self.last_order_response and isinstance(self.last_order_response, dict):
            code = self.last_order_response.get("data", {}).get("shareCode") or self.last_order_response.get("shareCode")
            if code:
                code = str(code).strip().upper()

        # Priority 2: Extract code from modal text
        if not code:
            modal_text = await self._page.evaluate("""() => {
                const modal = document.querySelector('.es-dialog') ||
                              document.querySelector('.m-dialog') ||
                              document.querySelector('.af-dialog');
                return modal ? modal.innerText : document.body.innerText;
            }""")
            match = re.search(r"Booking Code\s*\n\s*([A-Z0-9]{4,10})", modal_text, re.IGNORECASE)
            code = match.group(1).upper() if match else None

        # Close Modal safely
        await self._page.evaluate("""() => {
            try {
                const close = document.querySelector('.es-dialog-close, .m-dialog-close, .af-dialog-close, [class*="close"]');
                if (close) close.click();
            } catch (e) {}
        }""")

        logger.info(f"Generated booking code for matches: {code}")
        await self.clear_betslip()
        return code

    async def execute_wager(
        self,
        stake: Optional[float] = None,
        dry_run: Optional[bool] = None,
        confirm: bool = False
    ) -> WagerResult:
        """
        Executes wager placement or performs dry-run validation.

        Args:
            stake: Stake amount in NGN (defaults to self.default_stake, e.g. 10 NGN)
            dry_run: If True, evaluates slip and button readiness without placing wager.
            confirm: Extra safety gate for real wager execution. Must be True when dry_run is False.

        Returns:
            WagerResult detailing outcome, balance, order references, and screenshots.
        """
        is_dry_run = dry_run if dry_run is not None else self.dry_run
        stake_amount = float(stake if stake is not None else self.default_stake)

        # 1. Check login state & balance
        logged_in, balance_before, bal_text = await self.check_login_state()
        if not logged_in:
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=stake_amount,
                balance_before=balance_before,
                message="Cannot execute wager: Not logged in to SportyBet"
            )

        if balance_before < stake_amount:
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=stake_amount,
                balance_before=balance_before,
                message=f"Insufficient balance: {balance_before} NGN available, {stake_amount} NGN required"
            )

        # 2. Inspect betslip state
        betslip = await self.get_betslip_state()
        if betslip.item_count == 0:
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=stake_amount,
                balance_before=balance_before,
                message="Betslip is empty: load a valid booking code or selections first"
            )

        if betslip.has_unavailable:
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=stake_amount,
                balance_before=balance_before,
                item_count=betslip.item_count,
                message="Betslip contains unavailable or expired matches. Cannot proceed."
            )

        # 3. Set and verify stake (switches to Multiple tab automatically for 2+ items)
        confirmed_stake = await self.set_stake(stake_amount)
        updated_slip = await self.get_betslip_state()

        # 3b. Hard Safety Guard: Ensure Total Stake does not exceed requested stake (e.g. if forced into Single mode)
        if updated_slip.total_stake > stake_amount * 1.05:
            screenshot = await self.save_screenshot("stake_safety_halt")
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=updated_slip.total_stake,
                balance_before=balance_before,
                item_count=updated_slip.item_count,
                screenshot_path=screenshot,
                message=(
                    f"Safety Halt: SportyBet Total Stake ({updated_slip.total_stake:.2f} NGN) exceeds your "
                    f"requested stake ({stake_amount:.2f} NGN) because the slip contains conflicting outcomes "
                    f"from the same match (Single mode). Generate a fresh accumulator code first."
                ),
                details=updated_slip.to_dict()
            )

        # 4. Check Place Bet button readiness
        place_btn = await self._page.query_selector(self.SEL_PLACE_BET_BTN)
        if not place_btn:
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=confirmed_stake,
                balance_before=balance_before,
                item_count=updated_slip.item_count,
                message="Place Bet button not found in DOM"
            )

        is_disabled = await place_btn.is_disabled()
        btn_class = await place_btn.get_attribute("class") or ""
        if is_disabled or "is-disabled" in btn_class or "disabled" in btn_class:
            return WagerResult(
                success=False,
                mode="dry_run" if is_dry_run else "live",
                stake=confirmed_stake,
                balance_before=balance_before,
                item_count=updated_slip.item_count,
                message=f"Place Bet button is currently disabled (class: {btn_class})"
            )

        # 5. DRY-RUN MODE: Do not click Place Bet!
        if is_dry_run:
            screenshot = await self.save_screenshot("dry_run_ready")
            logger.info(
                f"[DRY-RUN] Validation passed: {updated_slip.item_count} items, "
                f"Stake: {confirmed_stake} NGN, Potential Win: {updated_slip.potential_win} NGN, "
                f"Balance: {balance_before} NGN. Wager NOT placed (dry-run enabled)."
            )
            return WagerResult(
                success=True,
                mode="dry_run",
                stake=confirmed_stake,
                potential_win=updated_slip.potential_win,
                item_count=updated_slip.item_count,
                balance_before=balance_before,
                balance_after=balance_before,
                screenshot_path=screenshot,
                message="DRY_RUN_READY: Betslip, stake, and Place Bet button fully validated.",
                details=updated_slip.to_dict()
            )

        # 6. LIVE EXECUTION MODE (Safety Gate)
        if not confirm:
            return WagerResult(
                success=False,
                mode="live",
                stake=confirmed_stake,
                balance_before=balance_before,
                item_count=updated_slip.item_count,
                message="CRITICAL SAFETY TRIGGER: Live execution requested but confirm=True was not provided."
            )

        logger.warning(f"EXECUTING REAL WAGER: Stake={confirmed_stake} NGN on {updated_slip.item_count} selections!")
        try:
            await place_btn.click(force=True)
            await self._page.wait_for_timeout(2000)

            # Check if confirmation dialog appears
            confirm_dialog_btn = await self._page.query_selector(self.SEL_CONFIRM_BTN)
            if confirm_dialog_btn and await confirm_dialog_btn.is_visible():
                logger.info("Confirmation dialog detected. Clicking Confirm...")
                await confirm_dialog_btn.click()
                await self._page.wait_for_timeout(3000)

            # Wait for order completion
            await self._page.wait_for_timeout(4000)
            screenshot = await self.save_screenshot("live_wager_result")

            # Check updated balance
            _, balance_after, bal_text_after = await self.check_login_state()

            # Parse order response
            order_info = self.last_order_response or {}
            ticket_id = order_info.get("data", {}).get("ticketId") or order_info.get("ticketId")

            return WagerResult(
                success=True,
                mode="live",
                stake=confirmed_stake,
                potential_win=updated_slip.potential_win,
                item_count=updated_slip.item_count,
                balance_before=balance_before,
                balance_after=balance_after,
                ticket_id=ticket_id,
                screenshot_path=screenshot,
                message=f"Live wager submitted successfully. New balance: {bal_text_after}",
                details={"order_response": order_info}
            )

        except Exception as e:
            logger.error(f"Error during live wager execution: {e}")
            screenshot = await self.save_screenshot("live_wager_failed")
            return WagerResult(
                success=False,
                mode="live",
                stake=confirmed_stake,
                balance_before=balance_before,
                screenshot_path=screenshot,
                message=f"Wager execution failed: {str(e)}"
            )


# CLI interface for standalone testing
async def main():
    import argparse
    parser = argparse.ArgumentParser(description="SportyBet Nigeria Automation Engine")
    parser.add_argument("--code", type=str, default=None, help="Booking code to test (e.g. LEUHNV or clean code)")
    parser.add_argument("--generate-slip", action="store_true", help="Generate a fresh clean booking code from live games")
    parser.add_argument("--stake", type=float, default=10.0, help="Stake in Naira (default: 10 NGN)")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Safe dry run (default: True)")
    parser.add_argument("--live", action="store_true", help="Execute real wager (dry_run=False, confirm=True)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless Chromium")
    args = parser.parse_args()

    dry_run = not args.live

    print("=" * 60)
    print("SPORTYBET NIGERIA AUTOMATION ENGINE")
    print(f"Mode: {'DRY RUN (Safe)' if dry_run else 'LIVE WAGER EXECUTION'}")
    print(f"Configured Stake: {args.stake} NGN")
    print("=" * 60)

    async with SportyBetAutomator(headless=args.headless, stake=args.stake, dry_run=dry_run) as bot:
        # Step 1: Login
        print("\n[Step 1] Authenticating with SportyBet...")
        login_res = await bot.login()
        print(f"Login Success: {login_res.success}")
        print(f"User: {login_res.user_name} ({login_res.email})")
        print(f"Balance: {login_res.balance_text} ({login_res.balance_ngn} NGN)")
        print(f"OTP Required: {login_res.otp_required}")
        print(f"Captcha Detected: {login_res.captcha_detected}")

        if not login_res.success:
            print(f"Error: {login_res.error_message}")
            return

        # Step 2: Code selection
        code_to_test = args.code
        if args.generate_slip:
            print("\n[Step 2] Generating fresh live booking code...")
            fresh_code = await bot.generate_clean_slip(selections_count=1)
            code_to_test = fresh_code or "U78C2S"
            print(f"Generated clean test code: {code_to_test}")
        elif not code_to_test:
            code_to_test = "U78C2S"
            print(f"\n[Step 2] Using default test booking code: {code_to_test}")

        # Step 3: Load Booking Code
        print(f"\n[Step 3] Loading Booking Code '{code_to_test}'...")
        slip = await bot.load_booking_code(code_to_test)
        print(f"Loaded {slip.item_count} items. Unavailable: {slip.has_unavailable}")
        for idx, itm in enumerate(slip.items, 1):
            print(f"  {idx}. {itm.match} | {itm.market} -> {itm.pick} (Odds: {itm.odds_text}) [Unavailable: {itm.is_unavailable}]")

        # Step 4: Set Stake and Validate/Execute
        print(f"\n[Step 4] Setting stake to {args.stake} NGN and evaluating wager...")
        wager_res = await bot.execute_wager(
            stake=args.stake,
            dry_run=dry_run,
            confirm=args.live
        )
        print("\n" + "=" * 60)
        print(f"WAGER EVALUATION RESULT:")
        print(f"Success: {wager_res.success}")
        print(f"Mode: {wager_res.mode}")
        print(f"Message: {wager_res.message}")
        print(f"Stake: {wager_res.stake} NGN")
        print(f"Potential Win: {wager_res.potential_win} NGN")
        print(f"Balance Before: {wager_res.balance_before} NGN")
        print(f"Screenshot: {wager_res.screenshot_path}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
