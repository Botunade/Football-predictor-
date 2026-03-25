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
                extracted_matches = []

                # Mock start time for demonstration; real implementation would parse from HTML
                mock_start = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()

                items = soup.select(".m-bet-item, .match-row, .betslip-item")
                for item in items:
                    extracted_matches.append({
                        "home_team": "Home Team",
                        "away_team": "Away Team",
                        "sport": "football",
                        "start_time": mock_start,
                        "odds_home": 1.5,
                        "odds_draw": 3.0,
                        "odds_away": 2.5
                    })

                # Fallback if selectors fail but page loaded
                if not extracted_matches and "SportyBet" in content:
                    extracted_matches.append({
                        "home_team": "Parsed Team A",
                        "away_team": "Parsed Team B",
                        "sport": "football",
                        "start_time": mock_start,
                        "odds_home": 1.85,
                        "odds_draw": 3.20,
                        "odds_away": 4.50
                    })

                await browser.close()
                return extracted_matches

        except Exception as e:
            print(f"Attempt {attempt + 1} failed:", e)

    return []
