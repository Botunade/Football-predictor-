import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time

load_dotenv()

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
BASE_URL_ODDS = "https://api.the-odds-api.com/v4/sports"

HEADERS_FOOTBALL = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': API_FOOTBALL_KEY
}

def fetch_data(league_id, season):
    """Alias for fetch_fixtures to match requested modularity."""
    return fetch_fixtures(league_id, season)

def fetch_fixtures(league_id, season):
    """Fetch upcoming fixtures for a league and season."""
    url = f"{BASE_URL_FOOTBALL}/fixtures"
    params = {"league": league_id, "season": season, "next": 50}
    try:
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=10)
        data = response.json()
        return data.get("response", [])
    except Exception as e:
        print(f"Error fetching fixtures: {e}")
        return []

# Cache for Understat scraping
SCRAPE_CACHE = {}

def scrape_understat_team(team_name, season):
    """
    Scrape advanced stats from Understat for a team with retry and cache.
    """
    cache_key = f"{team_name}_{season}"
    if cache_key in SCRAPE_CACHE:
        # Check if cache is fresh (less than 12 hours)
        data, timestamp = SCRAPE_CACHE[cache_key]
        if time.time() - timestamp < 12 * 3600:
            return data

    # Basic mapping for known differences
    mapping = {
        "Manchester United": "Manchester_United",
        "Manchester City": "Manchester_City",
        "Tottenham Hotspur": "Tottenham",
        "Newcastle United": "Newcastle_United",
        "Chelsea": "Chelsea",
        "Arsenal": "Arsenal",
        "Liverpool": "Liverpool",
        "Leicester": "Leicester"
    }
    search_name = mapping.get(team_name, team_name.replace(" ", "_"))
    url = f"https://understat.com/team/{search_name}/{season}"

    for attempt in range(3): # 3 retry attempts
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Understat stores data in script tags as JSON strings
                import re
                import json
                scripts = soup.find_all('script')
                for s in scripts:
                    if s.string and 'statisticsData' in s.string:
                        json_text = re.search(r"JSON\.parse\('(.+?)'\)", s.string).group(1)
                        # Handle escapes in the JSON string
                        json_text = json_text.encode('utf-8').decode('unicode_escape')
                        data = json.loads(json_text)
                        # Extract some metrics (this is a simplified example)
                        # Real Understat data structure is deep
                        result = {
                            "xG": data.get("xG", 1.5),
                            "xGA": data.get("xGA", 1.2),
                            "xGD": data.get("xG", 1.5) - data.get("xGA", 1.2),
                            "NPxG": data.get("npxG", 1.4),
                            "PPDA": data.get("ppda", 10.5),
                            "possession": 50.0
                        }
                        SCRAPE_CACHE[cache_key] = (result, time.time())
                        return result

            # If not found in scripts, wait and retry
            time.sleep(2)
        except Exception as e:
            print(f"Scraping error for {team_name} (Attempt {attempt+1}): {e}")
            time.sleep(2)

    # Fallback to defaults if scraping fails
    return {
        "xG": 1.5, "xGA": 1.2, "xGD": 0.3, "NPxG": 1.4, "PPDA": 10.5, "possession": 50.0
    }

def fetch_player_info(team_id, league_id=39, season=2024):
    """Fetch injury, fatigue, and dependency metrics from API-Football."""
    # 1. Fetch injuries
    injuries_url = f"{BASE_URL_FOOTBALL}/injuries"
    injuries_params = {"team": team_id, "league": league_id, "season": season}

    injury_count = 0
    try:
        resp = requests.get(injuries_url, headers=HEADERS_FOOTBALL, params=injuries_params, timeout=10)
        injuries_data = resp.json().get("response", [])
        injury_count = len(injuries_data)
    except Exception as e:
        print(f"Error fetching injuries for team {team_id}: {e}")

    # 2. Fetch player stats for fatigue (example: minutes played in last 5 games)
    # This is a simplified version; real fatigue index would be more complex
    # Mocking the calculation for now but providing the structure
    fatigue_index = min(injury_count * 0.1, 1.0) # Placeholder logic

    return {
        "injury_index": min(injury_count / 11.0, 1.0), # Ratio to full squad
        "fatigue_index": fatigue_index,
        "key_player_dependency": 0.5 # Default
    }

def fetch_odds(sport="soccer_epl", regions="uk"):
    """Fetch odds from OddsAPI."""
    url = f"{BASE_URL_ODDS}/{sport}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "decimal"
    }
    response = requests.get(url, params=params)
    data = response.json()

    odds_list = []
    for match in data:
        home_team = match["home_team"]
        away_team = match["away_team"]
        # Taking the first bookmaker's odds for simplicity
        if match["bookmakers"]:
            outcomes = match["bookmakers"][0]["markets"][0]["outcomes"]
            h_odds = next(o["price"] for o in outcomes if o["name"] == home_team)
            a_odds = next(o["price"] for o in outcomes if o["name"] == away_team)
            d_odds = next(o["price"] for o in outcomes if o["name"] == "Draw")
            odds_list.append({
                "home_team": home_team,
                "away_team": away_team,
                "odds_home": h_odds,
                "odds_away": a_odds,
                "odds_draw": d_odds
            })
    return odds_list

def compute_context(fixture):
    """Compute contextual metrics like rest days, derby flag, etc."""
    # Placeholders for more granular metrics
    return {
        "home_away_advantage": 1.1,
        "rest_days_diff": 0,
        "is_derby": 0,
        "weather_impact": 0,
        "pitch_quality": 1,
        "travel_distance": 50, # km
        "match_importance": 0.8 # 0 to 1
    }

def fetch_market_data(home_team, away_team):
    """Fetch market signals like betting volume and sentiment."""
    # Placeholder for OddsAPI or other specialized betting volume API
    return {
        "line_movement": 0.05,
        "public_sentiment": 0.6, # 60% on home
        "betting_volume": 1000000
    }

def build_features(fixture):
    """Extract and build features for a single fixture."""
    home_id = fixture["teams"]["home"]["id"]
    away_id = fixture["teams"]["away"]["id"]
    home_name = fixture["teams"]["home"]["name"]
    away_name = fixture["teams"]["away"]["name"]
    league_id = fixture["league"]["id"]
    season = fixture["league"]["season"]

    home_stats = scrape_understat_team(home_name, season)
    away_stats = scrape_understat_team(away_name, season)

    home_players = fetch_player_info(home_id, league_id, season)
    away_players = fetch_player_info(away_id, league_id, season)

    context = compute_context(fixture)

    return {
        "fixture_id": fixture["fixture"]["id"],
        "home_team": home_name,
        "away_team": away_name,
        "home_xG": home_stats["xG"],
        "away_xG": away_stats["xG"],
        "home_xGA": home_stats["xGA"],
        "away_xGA": away_stats["xGA"],
        "home_xGD": home_stats["xGD"],
        "away_xGD": away_stats["xGD"],
        "home_ppda": home_stats["PPDA"],
        "away_ppda": away_stats["PPDA"],
        "home_injury": home_players["injury_index"],
        "away_injury": away_players["injury_index"],
        "home_fatigue": home_players["fatigue_index"],
        "away_fatigue": away_players["fatigue_index"],
        **context
    }

def build_dataset(fixtures):
    """Merge all data sources into a single structured DataFrame."""
    rows = []
    for fix in fixtures:
        row = build_features(fix)
        rows.append(row)

    return pd.DataFrame(rows)
