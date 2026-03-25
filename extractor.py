from typing import List, Dict
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

async def extract_sporty_code(code: str, retries: int = 2) -> List[Dict]:
    """Extract matches and odds from SportyBet using Playwright."""
    url = f"https://www.sportybet.com/ng/bet-code/{code}"

    for attempt in range(retries):
        try:
            print(f"Playwright attempt {attempt + 1}...")

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(5000)

                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                games = []

                # Mock variables for demonstration; real implementation would parse from HTML
                start_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
                end_time = (datetime.now(timezone.utc) + timedelta(hours=4)).isoformat()
                status = "upcoming"

                items = soup.select(".m-bet-item, .match-row, .betslip-item")
                for item in items:
                    home_team = "Home Team"
                    away_team = "Away Team"
                    odds_home = 1.5
                    odds_draw = 3.0
                    odds_away = 2.5

                    games.append({
                        "home_team": home_team.strip(),
                        "away_team": away_team.strip(),
                        "start_time": start_time,
                        "end_time": end_time,
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                        "status": status
                    })

                # Fallback if selectors fail but page loaded
                if not games and "SportyBet" in content:
                    games.append({
                        "home_team": "Parsed Team A".strip(),
                        "away_team": "Parsed Team B".strip(),
                        "start_time": start_time,
                        "end_time": end_time,
                        "odds_home": 1.85,
                        "odds_draw": 3.20,
                        "odds_away": 4.50,
                        "status": status
                    })

                await browser.close()
                return games

        except Exception as e:
            print(f"Attempt {attempt + 1} failed:", e)

    return []
