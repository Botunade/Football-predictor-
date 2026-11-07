"""
SportyBet Nigeria Automation Module
Provides automated session management, authentication, betslip manipulation,
stake configuration, dry-run validation, and wager execution on SportyBet Nigeria.
"""

import asyncio
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
        """Removes all current selections from the betslip"""
        if not self._page:
            return False

        try:
            items_count = await self._page.evaluate(
                "() => document.querySelectorAll('.m-betslips .m-item, .m-item').length"
            )
            if items_count == 0:
                logger.debug("Betslip is already empty.")
                return True

            logger.info(f"Clearing {items_count} items from betslip...")
            # Click all delete icons and Remove All via JS eval
            await self._page.evaluate("""() => {
                const deletes = document.querySelectorAll('.m-icon-delete');
                deletes.forEach(d => d.click());
                const removeAll = document.querySelector('.m-betslips .m-text-min');
                if (removeAll) removeAll.click();
            }""")
            await self._page.wait_for_timeout(1000)

            # Confirm if modal appears
            await self._page.evaluate("""() => {
                const btn = Array.from(document.querySelectorAll('button, .af-button')).find(
                    e => e.innerText && ['Confirm', 'Yes'].includes(e.innerText.trim())
                );
                if (btn) btn.click();
            }""")
            await self._page.wait_for_timeout(1000)

            remaining = await self._page.evaluate(
                "() => document.querySelectorAll('.m-betslips .m-item, .m-item').length"
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

            # Read stake and potential win
            stake_val = 0.0
            pot_win = 0.0
            stake_inp = await self._page.query_selector(self.SEL_STAKE_INPUT)
            if stake_inp:
                inp_val = await stake_inp.input_value()
                try:
                    stake_val = float(inp_val.replace(",", ""))
                except Exception:
                    pass

            win_text = await self._page.eval_on_selector(
                ".m-stake",
                """
                el => {
                    const text = el.innerText || '';
                    const match = text.match(/Potential Win[\\s\\S]*?([\\d,]+\\.\\d{2})/);
                    return match ? match[1] : null;
                }
                """
            )
            if win_text:
                try:
                    pot_win = float(win_text.replace(",", ""))
                except Exception:
                    pass

            return BetslipState(
                item_count=len(parsed_items),
                items=parsed_items,
                has_unavailable=has_unavailable,
                total_stake=stake_val,
                potential_win=pot_win,
                can_place_bet=can_place
            )

        except Exception as e:
            logger.error(f"Failed to inspect betslip state: {e}")
            return BetslipState(error=str(e))

    async def set_stake(self, amount: Optional[float] = None) -> float:
        """
        Sets the stake value in the betslip (default: 10.0 NGN).
        Returns the confirmed stake amount.
        """
        stake_amount = float(amount if amount is not None else self.default_stake)
        if not self._page:
            return 0.0

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

    async def generate_clean_slip(self, selections_count: int = 1) -> Optional[str]:
        """
        Picks active live/upcoming matches on SportyBet, books the slip,
        and returns a fresh, valid booking code for testing.
        """
        if not self._page:
            await self.navigate_home()

        logger.info(f"Generating clean test booking code with {selections_count} selections...")
        await self.clear_betslip()

        odds_elements = await self._page.query_selector_all("span.m-outcome-odds")
        if not odds_elements:
            logger.warning("No odds elements found on page to generate slip.")
            return None

        clicked = 0
        for el in odds_elements:
            try:
                if await el.is_visible():
                    txt = (await el.inner_text()).strip()
                    odds_num = float(txt.replace(",", ""))
                    if 1.10 <= odds_num <= 10.0:
                        await el.click()
                        clicked += 1
                        await self._page.wait_for_timeout(1000)
                        if clicked >= selections_count:
                            break
            except Exception:
                continue

        if clicked == 0:
            logger.warning("Could not select any active odds.")
            return None

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

        await self._page.wait_for_timeout(3000)

        # Extract booking code from modal
        modal_text = await self._page.evaluate("""() => {
            const modal = document.querySelector('.es-dialog') ||
                          document.querySelector('.m-dialog') ||
                          document.querySelector('.af-dialog');
            return modal ? modal.innerText : document.body.innerText;
        }""")

        match = re.search(r"Booking Code\s*\n\s*([A-Z0-9]{5,8})", modal_text, re.IGNORECASE)
        code = match.group(1).upper() if match else None

        # Close dialog
        await self._page.evaluate("""() => {
            const close = document.querySelector('.es-dialog-close, .m-dialog-close, button:has-text("X"), .af-dialog-close');
            if (close) close.click();
        }""")
        await self._page.wait_for_timeout(500)

        logger.info(f"Generated clean booking code: {code}")
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

        # 3. Set and verify stake
        confirmed_stake = await self.set_stake(stake_amount)
        updated_slip = await self.get_betslip_state()

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
