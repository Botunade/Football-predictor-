import time
import random
from playwright.sync_api import sync_playwright

def extract_booking_code_data(code: str, retries: int = 2) -> list:
    """
    Extracts match data and odds from SportyBet using a booking code with retries.
    """
    for attempt in range(retries + 1):
        try:
            return _execute_extraction(code)
        except Exception as e:
            print(f"Extraction attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(random.uniform(5, 10))
    return []

def _execute_extraction(code: str) -> list:
    results = []
    with sync_playwright() as p:
        # Launch Chromium with anti-detection flags
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            # 1. Open SportyBet Nigeria
            page.goto("https://www.sportybet.com/ng", timeout=60000)
            time.sleep(random.uniform(3, 6)) # Randomized delay

            # 2. Locate and fill Booking Code input
            # Use placeholder selector as requested
            input_selector = 'input[placeholder="Booking Code"]'
            page.wait_for_selector(input_selector, timeout=20000)
            page.fill(input_selector, code)

            # 3. Click Load button
            load_button = page.get_by_role("button", name="Load")
            load_button.click()

            # 4. Wait for betslip to render
            # Stable selector for matches in betslip
            time.sleep(random.uniform(4, 7))

            # SportyBet structure: matches are usually in a list within the betslip container
            # We'll look for containers that hold team names and odds
            matches = page.locator('.m-bet-item').all() # Example stable class or similar

            if not matches:
                # Fallback to more generic betslip items if .m-bet-item fails
                matches = page.locator('.betslip-item').all()

            for match in matches:
                try:
                    # Extract Teams - SportyBet structure can be slightly nested
                    teams_text = match.locator('.m-team-name').all_text_contents()
                    if len(teams_text) < 2:
                         # Try searching deeper or alternate classes
                         teams_text = match.locator('div > span').all_text_contents()

                    if len(teams_text) >= 2:
                        home, away = teams_text[0].strip(), teams_text[1].strip()
                    else:
                        continue

                    # SportyBet often uses abbreviated names, normalization would go here

                    # Extract Odds
                    # SportyBet displays odds in specific fields
                    odds_elements = match.locator('.m-outcome-odds').all_text_contents()
                    # Market detection (1X2, Over/Under, etc.)
                    market_text = match.locator('.m-market-name').inner_text().strip()

                    odds_h = 0.0
                    odds_d = None
                    odds_a = 0.0

                    # Detection for H2H/1X2 market
                    if "1X2" in market_text or "Home/Away" in market_text:
                        if len(odds_elements) >= 3:
                            odds_h = float(odds_elements[0])
                            odds_d = float(odds_elements[1])
                            odds_a = float(odds_elements[2])
                        elif len(odds_elements) == 2:
                            odds_h = float(odds_elements[0])
                            odds_a = float(odds_elements[1])
                            odds_d = None # Likely basketball or similar

                    sport = "football"
                    if odds_d is None: sport = "basketball" # Simple heuristic

                    results.append({
                        "sport": sport,
                        "home": home,
                        "away": away,
                        "market": market_text,
                        "odds_home": odds_h,
                        "odds_draw": odds_d,
                        "odds_away": odds_a
                    })
                except Exception as e:
                    print(f"Error parsing match item: {e}")
                    continue

        except Exception as e:
            print(f"Playwright error for code {code}: {e}")
        finally:
            browser.close()

    return results
