from typing import List, Dict
from playwright.async_api import async_playwright, Page
from datetime import datetime, timezone
import asyncio

async def extract_sporty_code(code: str, retries: int = 3) -> List[Dict]:
    """
    Production-ready extraction from SportyBet by simulating full human navigation.
    """
    for attempt in range(retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                # Use a real user agent to avoid bot detection
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
                )
                page: Page = await context.new_page()

                print(f"[Extractor] Attempt {attempt+1}: Navigating to SportyBet homepage...")
                await page.goto("https://www.sportybet.com/ng/", timeout=60000)
                await page.wait_for_timeout(5000)  # wait for JS load

                # 1. Open BetSlip panel
                print("[Extractor] Opening betslip panel...")
                betslip_selectors = [
                    "button[data-testid='load-betslip']",
                    ".betslip-btn",
                    ".m-betslip-btn",
                    "button:has-text('BetSlip')",
                    ".m-betslip"
                ]

                betslip_btn = None
                for selector in betslip_selectors:
                    btn = await page.query_selector(selector)
                    if btn:
                        betslip_btn = btn
                        print(f"[Extractor] Found betslip button with selector: {selector}")
                        break

                if betslip_btn:
                    await betslip_btn.click()
                    await page.wait_for_timeout(2000)
                else:
                    print("[Extractor] Warning: Betslip button not found, checking if panel is already open.")

                # 2. Input the booking code
                print(f"[Extractor] Entering code: {code}")
                input_selectors = [
                    "input[data-testid='betslip-code-input']",
                    "input[placeholder*='Booking Code']",
                    ".m-betslip-input",
                    "input.m-input"
                ]

                code_input = None
                for selector in input_selectors:
                    inp = await page.query_selector(selector)
                    if inp:
                        code_input = inp
                        print(f"[Extractor] Found input field with selector: {selector}")
                        break

                if not code_input:
                    print("[Extractor] Error: Booking input field not found.")
                    # Take a screenshot for debugging (if possible in this env)
                    await browser.close()
                    continue

                await code_input.fill(code)
                await page.wait_for_timeout(1000)

                # 3. Click Load
                load_selectors = [
                    "button[data-testid='betslip-submit']",
                    "button:has-text('Load')",
                    ".m-betslip-submit",
                    ".m-btn-load"
                ]

                load_btn = None
                for selector in load_selectors:
                    btn = await page.query_selector(selector)
                    if btn:
                        load_btn = btn
                        print(f"[Extractor] Found load button with selector: {selector}")
                        break

                if load_btn:
                    await load_btn.click()
                    print("[Extractor] Load clicked, waiting for matches...")
                    await page.wait_for_timeout(5000)
                else:
                    print("[Extractor] Error: Load button not found.")
                    await browser.close()
                    continue

                # 4. Scrape games
                # Look for data-testid elements or common React class patterns
                print("[Extractor] Scoping matches...")

                # Check for error message
                error_msg = await page.query_selector(".m-error-msg, :has-text('Invalid code'), :has-text('expired')")
                if error_msg:
                    print(f"[Extractor] SportyBet reports: {await error_msg.inner_text()}")
                    await browser.close()
                    return []

                # Find all match containers
                items = await page.query_selector_all(".betslip-item, .m-bet-item, .match-row, div[data-testid*='match']")

                if not items:
                    print("[Extractor] No match items found with standard selectors. Attempting fallback scraping...")
                    # Fallback: check for any element containing 'vs' or team-like structures
                    items = await page.query_selector_all(".m-betslip-item")

                games = []
                for i, item in enumerate(items):
                    try:
                        # Improved Team Selectors
                        home_el = await item.query_selector(".home-team, .m-team-name, .home-team-name, span:nth-child(1)")
                        away_el = await item.query_selector(".away-team, .m-team-name:last-child, .away-team-name, span:nth-child(2)")

                        home_team = (await home_el.inner_text()).strip() if home_el else f"Home_{i+1}"
                        away_team = (await away_el.inner_text()).strip() if away_el else f"Away_{i+1}"

                        # Improved Odds Selectors
                        odds_els = await item.query_selector_all(".odds, .m-odds-num, .m-bet-odds")
                        if len(odds_els) >= 2:
                            odds_home = float((await odds_els[0].inner_text()).strip())
                            # Handle draw if exists, else 0.0
                            odds_draw = float((await odds_els[1].inner_text()).strip()) if len(odds_els) > 2 else 0.0
                            odds_away = float((await odds_els[-1].inner_text()).strip())
                        else:
                            odds_home, odds_draw, odds_away = 1.5, 3.0, 2.5 # Defaults

                        # Timestamps
                        start_time = await item.get_attribute("data-start-time") or datetime.now(timezone.utc).isoformat()

                        games.append({
                            "home_team": home_team,
                            "away_team": away_team,
                            "odds_home": odds_home,
                            "odds_draw": odds_draw,
                            "odds_away": odds_away,
                            "start_time": start_time,
                            "status": "upcoming"
                        })
                    except Exception as e:
                        print(f"[Extractor] Error parsing item {i}: {e}")

                if games:
                    print(f"[Extractor] Successfully extracted {len(games)} games.")
                    await browser.close()
                    return games
                else:
                    print("[Extractor] No games parsed from items.")
                    # Log the page content for inspection if no games found
                    content = await page.content()
                    print(f"[Extractor] Page content snippet: {content[:500]}...")

                await browser.close()

        except Exception as e:
            print(f"[Extractor Attempt {attempt+1}] Critical error: {e}")
            await asyncio.sleep(2)

    print("[Extractor] All attempts failed. No games extracted.")
    return []
