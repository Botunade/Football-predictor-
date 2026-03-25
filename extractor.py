from typing import List, Dict
from playwright.async_api import async_playwright, Page
from datetime import datetime, timezone
import asyncio

async def extract_sporty_code(code: str, retries: int = 2) -> List[Dict]:
    """
    Extract matches and odds from SportyBet by simulating human navigation.
    """
    for attempt in range(retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page: Page = await browser.new_page()

                # 1. Go to homepage
                await page.goto("https://www.sportybet.com/ng/", timeout=60000)
                await page.wait_for_timeout(5000)  # allow full JS load

                # 2. Open betslip panel
                # (Update selector based on actual site button)
                betslip_button = await page.query_selector("button[data-testid='load-betslip']")
                if betslip_button:
                    await betslip_button.click()
                    await page.wait_for_timeout(2000)

                # 3. Enter the code
                code_input = await page.query_selector("input[data-testid='betslip-code-input']")
                submit_button = await page.query_selector("button[data-testid='betslip-submit']")
                if not code_input or not submit_button:
                    print("[Extractor] Betslip input or submit not found!")
                    # Optional: fallback to direct URL if data-testid not present
                    await page.goto(f"https://www.sportybet.com/ng/bet-code/{code}", timeout=60000)
                    await page.wait_for_timeout(5000)
                else:
                    await code_input.fill(code)
                    await submit_button.click()
                    await page.wait_for_timeout(5000)  # wait for matches to load

                # 4. Scrape matches
                items = await page.query_selector_all(".betslip-item, .m-bet-item, .match-row")
                games = []

                for item in items:
                    # Teams
                    home_team_el = await item.query_selector(".home-team, .home-team-name, .m-team-name")
                    away_team_el = await item.query_selector(".away-team, .away-team-name, .m-team-name:last-child")
                    home_team = await home_team_el.inner_text() if home_team_el else "Home Team"
                    away_team = await away_team_el.inner_text() if away_team_el else "Away Team"

                    # Odds
                    odds_els = await item.query_selector_all(".odds-home, .odds-draw, .odds-away, .m-odds-num")
                    try:
                        odds_home = float(await odds_els[0].inner_text()) if len(odds_els) > 0 else 1.5
                        odds_draw = float(await odds_els[1].inner_text()) if len(odds_els) > 1 else 3.0
                        odds_away = float(await odds_els[2].inner_text()) if len(odds_els) > 2 else 2.5
                    except:
                        odds_home, odds_draw, odds_away = 1.5, 3.0, 2.5

                    # Timestamps
                    start_time = await item.get_attribute("data-start-time") or datetime.now(timezone.utc).isoformat()
                    end_time = await item.get_attribute("data-end-time") or datetime.now(timezone.utc).isoformat()

                    games.append({
                        "home_team": home_team.strip(),
                        "away_team": away_team.strip(),
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                        "start_time": start_time,
                        "end_time": end_time,
                        "status": "upcoming"
                    })

                await browser.close()
                return games

        except Exception as e:
            print(f"[Extractor Attempt {attempt+1}] Error: {e}")

    return []
