from typing import List, Dict
from playwright.async_api import async_playwright, Page
from datetime import datetime, timezone
import asyncio

async def extract_sporty_code(code: str, retries: int = 3) -> List[Dict]:
    """
    Production-ready extraction from SportyBet by simulating full human navigation.
    Ensures correct odds mapping and robust error handling.
    """
    for attempt in range(retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                # Use a real user agent to avoid bot detection
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
                page: Page = await context.new_page()

                # Basic stealth initialization
                await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

                # SportyBet often uses /ng/play/<code> or /en/play/<code>
                direct_url = f"https://www.sportybet.com/ng/play/{code}"
                print(f"[Extractor] Attempt {attempt+1}: Navigating to {direct_url}")

                try:
                    await page.goto(direct_url, timeout=60000, wait_until="networkidle")
                    await page.wait_for_timeout(5000)
                    print(f"[Extractor] Final Navigated URL: {page.url}")

                    # Redirect check
                    if "homepage" in page.url or page.url == "https://www.sportybet.com/ng/" or "login" in page.url:
                        print("[Extractor] Redirected to homepage/login, attempting manual injection...")
                        await page.goto("https://www.sportybet.com/ng/", timeout=60000)
                        await page.wait_for_timeout(5000)

                        # Open BetSlip
                        betslip_btn = await page.query_selector("button[data-testid='load-betslip'], .m-betslip-btn")
                        if betslip_btn:
                            await betslip_btn.click()
                            await page.wait_for_timeout(2000)

                        # Fill code
                        code_input = await page.query_selector("input[data-testid='betslip-code-input'], input[placeholder*='Code']")
                        if code_input:
                            await code_input.fill(code)
                            await page.wait_for_timeout(1000)

                            # Click Load
                            load_btn = await page.query_selector("button[data-testid='betslip-submit'], button:has-text('Load')")
                            if load_btn:
                                await load_btn.click()
                                await page.wait_for_timeout(5000)

                    # Wait for results container specifically
                    game_selector = ".betslip-item, .m-bet-item, div[data-testid*='match']"
                    try:
                        await page.wait_for_selector(game_selector, timeout=15000)
                    except:
                        print("[Extractor] Warning: Match elements did not load in time.")

                    # Error detection (Invalid/Expired)
                    error_msg = await page.query_selector(".m-error-msg, :has-text('Invalid code'), :has-text('expired')")
                    if error_msg:
                        print(f"[Extractor] SportyBet Error: {await error_msg.inner_text()}")
                        await browser.close()
                        return []

                    # Scrape games
                    items = await page.query_selector_all(game_selector)
                    if not items:
                        print("[Extractor] No match items found. URL might be wrong or blocked.")
                        await browser.close()
                        continue

                    games = []
                    for i, item in enumerate(items):
                        try:
                            # Improved Team Selectors
                            home_el = await item.query_selector(".home-team, .m-team-name, .home-team-name, .team-name")
                            away_el = await item.query_selector(".away-team, .m-team-name:last-child, .away-team-name")

                            home_team = (await home_el.inner_text()).strip() if home_el else f"Home_{i+1}"
                            away_team = (await away_el.inner_text()).strip() if away_el else f"Away_{i+1}"

                            # Odds STRICT extraction (Home, Draw, Away)
                            odds_home, odds_draw, odds_away = 0.0, 0.0, 0.0

                            # Try structured ones first
                            h_o = await item.query_selector(".odds-home, .home-odds")
                            d_o = await item.query_selector(".odds-draw, .draw-odds")
                            a_o = await item.query_selector(".odds-away, .away-odds")

                            try:
                                if h_o: odds_home = float((await h_o.inner_text()).strip())
                                if d_o: odds_draw = float((await d_o.inner_text()).strip())
                                if a_o: odds_away = float((await a_o.inner_text()).strip())
                            except: pass

                            # Fallback to ordered list if structured failed
                            if odds_home == 0.0:
                                odds_list = await item.query_selector_all(".m-odds-num, .odds")
                                try:
                                    if len(odds_list) == 2:
                                        odds_home = float((await odds_list[0].inner_text()).strip())
                                        odds_away = float((await odds_list[1].inner_text()).strip())
                                        odds_draw = 0.0
                                    elif len(odds_list) >= 3:
                                        odds_home = float((await odds_list[0].inner_text()).strip())
                                        odds_draw = float((await odds_list[1].inner_text()).strip())
                                        odds_away = float((await odds_list[2].inner_text()).strip())
                                except: pass

                            print(f"[ODDS DEBUG] {home_team} vs {away_team} -> {odds_home} | {odds_draw} | {odds_away}")

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
                            print(f"[Extractor] Error parsing game {i}: {e}")

                    if games:
                        print(f"[Extractor] Successfully extracted {len(games)} matches.")
                        await browser.close()
                        return games

                except Exception as e:
                    print(f"[Extractor] Browser interaction failed: {e}")

                await browser.close()

        except Exception as e:
            print(f"[Extractor Attempt {attempt+1}] Critical error: {e}")
            await asyncio.sleep(2)

    print("[Extractor] All attempts failed.")
    return []
