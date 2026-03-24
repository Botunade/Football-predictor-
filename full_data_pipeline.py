import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import time

load_dotenv()

# Single key for all sports
API_KEY = os.getenv("API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

SPORTS_CONFIG = {
    "football": {
        "base_url": "https://v3.football.api-sports.io",
        "api_key": API_KEY,
        "host": "v3.football.api-sports.io"
    },
    "basketball": {
        "base_url": "https://v1.basketball.api-sports.io",
        "api_key": API_KEY,
        "host": "v1.basketball.api-sports.io"
    },
    "hockey": {
        "base_url": "https://v1.hockey.api-sports.io",
        "api_key": API_KEY,
        "host": "v1.hockey.api-sports.io"
    }
}

BASE_URL_ODDS = "https://api.the-odds-api.com/v4/sports"

# Reliable scraping URL templates from .env or fallback
SCRAPE_URLS = {
    "football": os.getenv("FOOTBALL_SCRAPE_URLS", "https://fbref.com/,https://understat.com/").split(","),
    "basketball": os.getenv("BASKETBALL_SCRAPE_URLS", "https://www.basketball-reference.com/").split(","),
    "hockey": os.getenv("HOCKEY_SCRAPE_URLS", "https://www.hockey-reference.com/").split(",")
}

def fetch_api_data(sport, endpoint, params=None):
    """Fetch data from API-Sports"""
    if sport not in SPORTS_CONFIG:
        print(f"Error: Sport {sport} not supported.")
        return None

    config = SPORTS_CONFIG[sport]
    url = f"{config['base_url']}/{endpoint}"
    headers = {"x-apisports-key": config["api_key"]}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching {sport} API data:", response.text)
            return None
    except Exception as e:
        print(f"Exception fetching {sport} data: {e}")
        return None

def fetch_data(league_id, season, sport="football"):
    """Alias for fetch_fixtures to match requested modularity."""
    return fetch_fixtures(league_id, season, sport)

def fetch_fixtures(league_id, season, sport="football"):
    """Fetch upcoming fixtures for a league and season for a given sport."""
    params = {"league": league_id, "season": season}
    if sport == "football":
        params["next"] = 50
    data = fetch_api_data(sport, "fixtures", params=params)
    return data.get("response", []) if data else []

def fetch_fixture_statistics(sport, fixture_id):
    """Fetch detailed statistics for a specific fixture."""
    data = fetch_api_data(sport, "fixtures/statistics", params={"fixture": fixture_id})
    return data.get("response", []) if data else []

def fetch_team_form(sport, league_id, season, team_id):
    """Fetch recent form statistics for a team."""
    data = fetch_api_data(sport, "teams/statistics", params={
        "league": league_id,
        "season": season,
        "team": team_id
    })
    return data.get("response", {}) if data else {}

def scrape_advanced_stats(sport, match_name):
    """
    Automatically try multiple sites for advanced stats.
    Returns a dictionary of key metrics.
    """
    stats = {}
    if sport not in SCRAPE_URLS:
        return stats

    for base_url in SCRAPE_URLS[sport]:
        try:
            # For simplicity, append match identifier to base_url
            url = f"{base_url}{match_name.replace(' ','-')}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                # Jules: implement site-specific parsing logic here
                # Example:
                # if "fbref.com" in base_url:
                #     stats['xG'] = ...
                pass
        except:
            continue
    return stats

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

def fetch_player_info(team_id, league_id=39, season=2024, sport="football"):
    """Fetch injury, fatigue, and dependency metrics for a given sport."""
    if sport != "football":
        # Placeholder for other sports
        return {"injury_index": 0.1, "fatigue_index": 0.2, "key_player_dependency": 0.5}

    # 1. Fetch injuries
    data = fetch_api_data("football", "injuries", params={"team": team_id, "league": league_id, "season": season})
    injuries_data = data.get("response", []) if data else []
    injury_count = len(injuries_data)

    # 2. Fetch player stats for fatigue (example: minutes played in last 5 games)
    # This is a simplified version; real fatigue index would be more complex
    # Mocking the calculation for now but providing the structure
    fatigue_index = min(injury_count * 0.1, 1.0) # Placeholder logic

    return {
        "injury_index": min(injury_count / 11.0, 1.0), # Ratio to full squad
        "fatigue_index": fatigue_index,
        "key_player_dependency": 0.5 # Default
    }

def fetch_odds(sport="football", league_id=39, regions="uk"):
    """Fetch odds from OddsAPI with sport mapping."""
    # OddsAPI sport mapping
    mapping = {
        "football": "soccer_epl", # Default to EPL, could be refined by league_id
        "basketball": "basketball_nba",
        "hockey": "icehockey_nhl"
    }
    odds_sport = mapping.get(sport, "soccer_epl")

    url = f"{BASE_URL_ODDS}/{odds_sport}/odds"
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

def build_features(fixture, sport="football", scraped_data=None):
    """
    Consolidated feature builder combining API data, scraped stats, and context.
    """
    """Extract and build features for a single fixture for a given sport."""
    # API-Sports structure varies slightly by sport, but fixtures usually have teams
    try:
        home_id = fixture["teams"]["home"]["id"]
        away_id = fixture["teams"]["away"]["id"]
        home_name = fixture["teams"]["home"]["name"]
        away_name = fixture["teams"]["away"]["name"]
        league_id = fixture["league"]["id"]
        season = fixture["league"]["season"]
        fixture_id = fixture["fixture"]["id"]
    except (KeyError, TypeError):
        # Fallback for manual or incomplete API structures
        home_name = fixture.get("home_team", fixture.get("home", "Home"))
        away_name = fixture.get("away_team", fixture.get("away", "Away"))
    except KeyError:
        # Fallback for manual or different API structures
        home_name = fixture.get("home_team", "Home")
        away_name = fixture.get("away_team", "Away")
        home_id = fixture.get("home_id", 0)
        away_id = fixture.get("away_id", 0)
        league_id = fixture.get("league_id", 0)
        season = fixture.get("season", 2024)
        fixture_id = fixture.get("fixture_id", 0)

    # 1. Fetch Basic Team Stats/Form from API
    home_api_stats = fetch_team_form(sport, league_id, season, home_id) if home_id else {}
    away_api_stats = fetch_team_form(sport, league_id, season, away_id) if away_id else {}

    # 2. Sport-Specific Advanced Data & Scraping
    if sport == "football":
        # Understat scraping
        home_scraped = scrape_understat_team(home_name, season)
        away_scraped = scrape_understat_team(away_name, season)

        features = {
            "home_xG": home_scraped.get("xG", 1.5),
            "away_xG": away_scraped.get("xG", 1.5),
            "home_xGA": home_scraped.get("xGA", 1.5),
            "away_xGA": away_scraped.get("xGA", 1.5),
            "home_ppda": home_scraped.get("PPDA", 10.0),
            "away_ppda": away_scraped.get("PPDA", 10.0),
            "home_possession": home_scraped.get("possession", 50.0),
            "away_possession": away_scraped.get("possession", 50.0),
        }
    elif sport == "basketball":
        features = {
            "home_points_avg": 110.0, # Placeholder for API points
            "away_points_avg": 108.0,
            "home_efficiency": 0.55,
            "away_efficiency": 0.52
        }
    else: # hockey
        features = {
            "home_goals_avg": 3.2,
            "away_goals_avg": 2.8,
            "home_saves_pct": 0.91,
            "away_saves_pct": 0.89
        }

    # 3. Player Info (Injuries, Fatigue)
    home_players = fetch_player_info(home_id, league_id, season, sport)
    away_players = fetch_player_info(away_id, league_id, season, sport)

    features.update({
        "home_injury": home_players["injury_index"],
        "away_injury": away_players["injury_index"],
        "home_fatigue": home_players["fatigue_index"],
        "away_fatigue": away_players["fatigue_index"]
    })

    # 4. Contextual Metrics (Rest, Derby, Weather)
    context = compute_context(fixture)
    features.update(context)

    # 5. Market Data (Line movement, sentiment)
    market = fetch_market_data(home_name, away_name)
    features.update(market)

    # 6. Override with any manual scraped_data
    if scraped_data:
        features.update(scraped_data)

    # Final Meta Data
    features.update({
        "fixture_id": fixture_id,
        "home_team": home_name,
        "away_team": away_name,
        "home_xGD": features.get("home_xG", 0) - features.get("home_xGA", 0),
        "away_xGD": features.get("away_xG", 0) - features.get("away_xGA", 0)
    })

    return features

    if sport == "football":
        home_stats = scrape_understat_team(home_name, season)
        away_stats = scrape_understat_team(away_name, season)
    else:
        # Basic stats for other sports from API fixtures if available
        # Example: recent goals/points
        home_stats = {"xG": 1.5, "xGA": 1.5, "xGD": 0, "PPDA": 10}
        away_stats = {"xG": 1.5, "xGA": 1.5, "xGD": 0, "PPDA": 10}

    # Add scraped_data if provided
    if scraped_data:
        home_stats.update(scraped_data.get("home", {}))
        away_stats.update(scraped_data.get("away", {}))

    home_players = fetch_player_info(home_id, league_id, season, sport)
    away_players = fetch_player_info(away_id, league_id, season, sport)

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

def build_dataset(fixtures, sport="football"):
    """Merge all data sources into a single structured DataFrame for a given sport."""
    rows = []
    for fix in fixtures:
        row = build_features(fix, sport)
        rows.append(row)

    return pd.DataFrame(rows)
