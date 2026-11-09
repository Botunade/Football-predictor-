"""
Multi-Sport Scraper & Modeling Engine
Expands the platform beyond football into Basketball (NBA, EuroLeague) and Tennis (ATP, WTA).
Features two-way moneyline, spreads/handicaps, and game/point totals.
"""

import json
import logging
import time
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MultiSportScraper")
logger.setLevel(logging.INFO)


class MultiSportScraper:
    """
    Crawls Basketball and Tennis fixtures and bookmaker odds from SportyBet.
    """

    BASKETBALL_URL = (
        "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents?"
        "sportId=sr%3Asport%3A2&"
        "marketId=1%2C18%2C16%2C200&"
        "pageSize=40&pageNum=1&option=1"
    )

    TENNIS_URL = (
        "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents?"
        "sportId=sr%3Asport%3A5&"
        "marketId=1%2C18%2C16%2C300&"
        "pageSize=40&pageNum=1&option=1"
    )

    def fetch_basketball_fixtures(self) -> List[Dict[str, Any]]:
        """
        Fetches live & upcoming Basketball fixtures (NBA, EuroLeague, etc.)
        """
        results = []
        ts = int(time.time() * 1000)
        url = f"{self.BASKETBALL_URL}&_t={ts}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for tourn in data.get("data", {}).get("tournaments", []):
                    league = tourn.get("name", "Basketball")
                    for ev in tourn.get("events", []):
                        h = ev.get("homeTeamName")
                        a = ev.get("awayTeamName")
                        if not h or not a:
                            continue

                        h_odd, a_odd = 1.90, 1.90
                        totals_dict = {}

                        for m in ev.get("markets", []):
                            m_id = str(m.get("id"))
                            if m_id in ["1", "200"]:  # Winner / Moneyline
                                for o in m.get("outcomes", []):
                                    oid = str(o.get("id"))
                                    val = float(o.get("odds") or 1.90)
                                    if oid == "1":
                                        h_odd = val
                                    elif oid == "2":
                                        a_odd = val
                            elif m_id == "18":  # Over/Under Points Total
                                spec = str(m.get("specifier", ""))
                                for o in m.get("outcomes", []):
                                    oid = str(o.get("id"))
                                    val = float(o.get("odds") or 1.90)
                                    if oid == "12":
                                        totals_dict[f"Over {spec}"] = val
                                    elif oid == "13":
                                        totals_dict[f"Under {spec}"] = val

                        results.append({
                            "sport": "basketball",
                            "home_team": h,
                            "away_team": a,
                            "league": league,
                            "odds_home": h_odd,
                            "odds_draw": 1.0,
                            "odds_away": a_odd,
                            "market_odds": totals_dict,
                            "sporty_id": str(ev.get("eventId", "")),
                            "recommended_outcome": "Home" if h_odd <= a_odd else "Away",
                            "recommended_odds": min(h_odd, a_odd)
                        })
        except Exception as e:
            logger.debug(f"Basketball fetch notice: {e}")

        logger.info(f"Fetched {len(results)} Basketball fixtures.")
        return results

    def fetch_tennis_fixtures(self) -> List[Dict[str, Any]]:
        """
        Fetches live & upcoming Tennis fixtures (ATP, WTA).
        """
        results = []
        ts = int(time.time() * 1000)
        url = f"{self.TENNIS_URL}&_t={ts}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for tourn in data.get("data", {}).get("tournaments", []):
                    league = tourn.get("name", "Tennis")
                    for ev in tourn.get("events", []):
                        p1 = ev.get("homeTeamName")
                        p2 = ev.get("awayTeamName")
                        if not p1 or not p2:
                            continue

                        p1_odd, p2_odd = 1.90, 1.90
                        for m in ev.get("markets", []):
                            if str(m.get("id")) in ["1", "300"]:
                                for o in m.get("outcomes", []):
                                    oid = str(o.get("id"))
                                    val = float(o.get("odds") or 1.90)
                                    if oid == "1":
                                        p1_odd = val
                                    elif oid == "2":
                                        p2_odd = val

                        results.append({
                            "sport": "tennis",
                            "home_team": p1,
                            "away_team": p2,
                            "league": league,
                            "odds_home": p1_odd,
                            "odds_draw": 1.0,
                            "odds_away": p2_odd,
                            "market_odds": {"Player 1": p1_odd, "Player 2": p2_odd},
                            "sporty_id": str(ev.get("eventId", "")),
                            "recommended_outcome": "Home" if p1_odd <= p2_odd else "Away",
                            "recommended_odds": min(p1_odd, p2_odd)
                        })
        except Exception as e:
            logger.debug(f"Tennis fetch notice: {e}")

        logger.info(f"Fetched {len(results)} Tennis fixtures.")
        return results
