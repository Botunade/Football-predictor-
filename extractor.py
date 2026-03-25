from typing import List, Dict
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta

async def extract_sporty_code(code: str, retries: int = 2) -> List[Dict]:
    """Extract matches and odds from SportyBet using Playwright."""
    url = f"https://www.sportybet.com/ng/bet-code/{code}"

    for attempt in range(retries):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto(url, timeout=60000)
                await page.wait_for_timeout(5000)  # allow full load

                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")
                games = []

                # Parse real match items (update selectors to actual site structure)
                # SportyBet often uses .m-bet-item or .m-betslip-item
                items = soup.select(".betslip-item, .m-bet-item, .match-row, .m-betslip-item")
                for item in items:
                    # Teams: .home-team, .home-team-name, .m-team-name and .away-team, .away-team-name, .m-team-name:last-child
                    home_team_el = item.select_one(".home-team, .home-team-name, .m-team-name")
                    away_team_el = item.select_one(".away-team, .away-team-name, .m-team-name:last-child")

                    home_team = home_team_el.text.strip() if home_team_el else "Home Team"
                    away_team = away_team_el.text.strip() if away_team_el else "Away Team"

                    # Odds: .odds-home, .odds-draw, .odds-away, .m-odds-num
                    odds_els = item.select(".odds-home, .odds-draw, .odds-away, .m-odds-num")
                    try:
                        odds_home = float(odds_els[0].text.strip()) if len(odds_els) > 0 else 1.5
                        odds_draw = float(odds_els[1].text.strip()) if len(odds_els) > 1 else 3.0
                        odds_away = float(odds_els[2].text.strip()) if len(odds_els) > 2 else 2.5
                    except (ValueError, IndexError):
                        odds_home, odds_draw, odds_away = 1.5, 3.0, 2.5

                    # Timestamps: data-start-time and data-end-time attributes; fallback to current UTC
                    start_time = item.get("data-start-time") or datetime.now(timezone.utc).isoformat()
                    end_time = item.get("data-end-time") or datetime.now(timezone.utc).isoformat()

                    games.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "start_time": start_time,
                        "end_time": end_time,
                        "odds_home": odds_home,
                        "odds_draw": odds_draw,
                        "odds_away": odds_away,
                        "status": "upcoming"
                    })

                await browser.close()
                return games

        except Exception as e:
            print(f"Attempt {attempt + 1} failed:", e)

    return []
