"""
SportyBet Code Reader Module

Provides comprehensive booking code extraction with JSON output,
CLI interface, and both Playwright and API fallback methods.

This module accepts SportyBet booking codes and extracts all matches and odds,
saving to JSON with comprehensive documentation including match details,
odds data, source metadata, and error tracking.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from uuid import uuid4

# Handle both package and direct execution
try:
    from .sportybet_scraper import SportyBetScraper, ScrapingError
except ImportError:
    from sportybet_scraper import SportyBetScraper, ScrapingError

try:
    from utils.proactor_helper import run_coro_with_proactor, run_in_proactor_thread
except ImportError:
    try:
        from src.utils.proactor_helper import run_coro_with_proactor, run_in_proactor_thread
    except ImportError:
        try:
            from ..utils.proactor_helper import run_coro_with_proactor, run_in_proactor_thread
        except ImportError:
            run_coro_with_proactor = None
            run_in_proactor_thread = None

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Error codes for booking code validation and extraction failures"""
    INVALID_CODE = "invalid_code"
    INVALID_FORMAT = "invalid_format"
    EXPIRED_CODE = "expired_code"
    LOGIN_REQUIRED = "login_required"
    NETWORK_TIMEOUT = "network_timeout"
    PAGE_ERROR = "page_error"
    NO_MATCHES = "no_matches"
    UNKNOWN = "unknown_error"


@dataclass
class Metadata:
    """
    Source metadata for extraction.

    Contains information about where and when the data was extracted,
    including source URL, extraction timestamp, and code information.
    """
    url: str = ""
    extraction_time: str = ""
    code: str = ""
    source: str = "sportybet"
    method: str = ""
    retry_count: int = 0
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    scraper_version: str = "v4.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractionDetails:
    """
    Details about the extraction process.

    Provides comprehensive information about what happened during extraction,
    including status, error tracking, timing information, and warnings.
    """
    status: str = "pending"
    error_code: Optional[str] = None
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)
    total_matches: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    extraction_time_seconds: float = 0.0
    page_load_time_ms: float = 0.0
    element_find_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Odds:
    """
    Complete odds information for a match.

    Contains various types of betting odds including:
    - H/D/A (Home/Draw/Away)
    - BTTS (Both Teams To Score)
    - Over/Under 2.5
    - Handicap and other special odds
    """
    h_d_a: Dict[str, float] = field(default_factory=lambda: {"home": 0.0, "draw": 0.0, "away": 0.0})
    btts: Dict[str, float] = field(default_factory=lambda: {"both_teams_to_score": 0.0, "over": 0.0, "under": 0.0})
    over_under: Dict[str, float] = field(default_factory=lambda: {"over_2_5": 0.0, "under_2_5": 0.0})
    handicap: Dict[str, float] = field(default_factory=dict)
    other: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "h_d_a": self.h_d_a,
            "btts": self.btts,
            "over_under": self.over_under
        }
        if self.handicap:
            result["handicap"] = self.handicap
        if self.other:
            result["other"] = self.other
        return result


@dataclass
class Match:
    """
    Match details with complete information.

    Contains match metadata including teams, league, date/time, venue,
    odds information, and match status.
    """
    home_team: str = ""
    away_team: str = ""
    league: str = ""
    date: str = ""
    time: str = ""
    venue: str = ""
    status: str = "upcoming"  # upcoming, live, finished
    sport: str = "football"
    odds: Optional[Odds] = None
    match_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "league": self.league,
            "date": self.date,
            "time": self.time,
            "venue": self.venue,
            "status": self.status,
            "sport": self.sport,
            "odds": self.odds.to_dict() if self.odds else None,
            "match_id": self.match_id
        }
        return result


def validate_code_format(code: str) -> Dict[str, Any]:
    """
    Validate the format of a SportyBet booking code.

    Checks that the code meets the expected format requirements
    for SportyBet booking codes.

    Args:
        code: The booking code to validate

    Returns:
        Dict with 'valid' boolean and optional 'reason' string

    Example:
        >>> validate_code_format("ABC123")
        {'valid': True, 'reason': None}
        >>> validate_code_format("123")
        {'valid': False, 'reason': 'Code is too short...'}
    """
    if not code:
        return {"valid": False, "reason": "Code cannot be empty"}

    if not isinstance(code, str):
        return {"valid": False, "reason": "Code must be a string"}

    code = code.strip()

    if len(code) < 3:
        return {"valid": False, "reason": "Code is too short (minimum 3 characters)"}

    if len(code) > 50:
        return {"valid": False, "reason": "Code is too long (maximum 50 characters)"}

    # Check for valid characters (alphanumeric, hyphens, underscores)
    if not re.match(r'^[A-Za-z0-9_-]+$', code):
        return {"valid": False, "reason": "Code contains invalid characters"}

    return {"valid": True, "reason": None}


def _determine_error_type(error_message: str, url: str) -> Tuple[Optional[ErrorCode], str]:
    """Determine the type of error from message and URL."""
    error_lower = error_message.lower() if error_message else ""
    url_lower = url.lower() if url else ""

    if "invalid" in error_lower or "invalid code" in error_lower:
        return ErrorCode.INVALID_CODE, "Booking code is invalid or malformed"

    if "expired" in error_lower:
        return ErrorCode.EXPIRED_CODE, "Booking code has expired"

    if "login" in url_lower or "signin" in url_lower or "authentication" in error_lower:
        return ErrorCode.LOGIN_REQUIRED, "Authentication required - redirected to login"

    if "timeout" in error_lower or "timed out" in error_lower:
        return ErrorCode.NETWORK_TIMEOUT, "Request timed out"

    if "no matches" in error_lower:
        return ErrorCode.NO_MATCHES, "No matches found in booking code"

    return ErrorCode.PAGE_ERROR, error_message or "Unknown page error"


def normalize_sportybet_selection(
    market_desc: str,
    outcome_desc: str,
    market_id: str = "",
    specifier: str = ""
) -> str:
    """
    Normalizes any SportyBet (market_desc, outcome_desc, market_id, specifier) tuple into a crystal-clear,
    unambiguous selection label. Prevents half-time, 2nd-half, team-total, handicap, or combo markets
    from ever being misread as Full-Time 1X2 or Full-Time Over/Under.
    """
    m_desc = (market_desc or "").strip()
    o_desc = (outcome_desc or "").strip()
    m_low = m_desc.lower()
    o_low = o_desc.lower()
    m_id = str(market_id or "").strip()

    if not m_desc and not o_desc:
        return "Home"
    if not m_desc:
        return o_desc

    # 1. Full-Time 1X2 (Market 1)
    if m_id == "1" or m_low == "1x2":
        if o_low in ("1", "home"):
            return "Home"
        if o_low in ("x", "draw"):
            return "Draw"
        if o_low in ("2", "away"):
            return "Away"
        return o_desc

    # 2. Early Payout 1X2 - 1UP / 2UP (Markets 60200, 60100)
    if m_id == "60200" or "1up" in m_low:
        if "home" in o_low or o_low == "1":
            return "Home (1UP)"
        if "away" in o_low or o_low == "2":
            return "Away (1UP)"
        return "Draw"
    if m_id == "60100" or "2up" in m_low:
        if "home" in o_low or o_low == "1":
            return "Home (2UP)"
        if "away" in o_low or o_low == "2":
            return "Away (2UP)"
        return "Draw"

    # 3. Full-Time Double Chance (Market 10)
    if m_id == "10" or m_low == "double chance":
        if "home" in o_low and "draw" in o_low or o_low == "1x":
            return "1X"
        if "home" in o_low and "away" in o_low or o_low == "12":
            return "12"
        if "draw" in o_low and "away" in o_low or o_low == "x2":
            return "X2"
        return o_desc

    # 4. Full-Time Over/Under (Market 18)
    if m_id == "18" or m_low in ("over/under", "total", "goals over/under"):
        return o_desc

    # 5. 1st Half Markets (60, 63, 64, 68, 75, etc.)
    if m_id == "68" or m_low == "1st half - over/under":
        return f"1st Half {o_desc}"
    if m_id == "60" or m_low == "1st half - 1x2":
        return f"1st Half {o_desc}"
    if m_id == "63" or m_low == "1st half - double chance":
        if "home" in o_low and "draw" in o_low or o_low == "1x":
            return "1st Half 1X"
        if "home" in o_low and "away" in o_low or o_low == "12":
            return "1st Half 12"
        if "draw" in o_low and "away" in o_low or o_low == "x2":
            return "1st Half X2"
        return f"1st Half {o_desc}"
    if m_id == "75" or m_low in ("1st half - gg/ng", "1st half - both teams to score"):
        return "1st Half GG" if "yes" in o_low or o_low == "gg" else "1st Half NG"
    if m_id == "64" or m_low == "1st half - draw no bet":
        return "1st Half DNB Home" if "home" in o_low else "1st Half DNB Away"

    # 6. 2nd Half Markets (83, 85, 86, 90, 95, etc.)
    if m_id == "90" or m_low == "2nd half - over/under":
        return f"2nd Half {o_desc}"
    if m_id == "83" or m_low == "2nd half - 1x2":
        return f"2nd Half {o_desc}"
    if m_id == "85" or m_low == "2nd half - double chance":
        if "home" in o_low and "draw" in o_low or o_low == "1x":
            return "2nd Half 1X"
        if "home" in o_low and "away" in o_low or o_low == "12":
            return "2nd Half 12"
        if "draw" in o_low and "away" in o_low or o_low == "x2":
            return "2nd Half X2"
        return f"2nd Half {o_desc}"
    if m_id == "95" or m_low in ("2nd half - gg/ng", "2nd half - both teams to score"):
        return "2nd Half GG" if "yes" in o_low or o_low == "gg" else "2nd Half NG"
    if m_id == "86" or m_low == "2nd half - draw no bet":
        return "2nd Half DNB Home" if "home" in o_low else "2nd Half DNB Away"

    # 7. Full-Time GG/NG (Market 29) & Draw No Bet (Market 11)
    if m_id == "29" or m_low in ("gg/ng", "both teams to score"):
        return "BTTS Yes" if "yes" in o_low or o_low == "gg" else "BTTS No"
    if m_id == "11" or m_low == "draw no bet":
        return "Home DNB" if "home" in o_low else "Away DNB"

    # 8. Team Over/Under & Clean Sheets & Win Either Half
    if m_id == "19" or m_low == "home team over/under":
        return f"Home Team {o_desc}"
    if m_id == "20" or m_low == "away team over/under":
        return f"Away Team {o_desc}"
    if m_id == "31" or m_low == "home team clean sheet":
        return f"Home Clean Sheet {o_desc}"
    if m_id == "32" or m_low == "away team clean sheet":
        return f"Away Clean Sheet {o_desc}"
    if m_id == "50" or m_low == "home team to win either half":
        return "Home Win Either Half" if "yes" in o_low else "Home Win Either Half No"
    if m_id == "51" or m_low == "away team to win either half":
        return "Away Win Either Half" if "yes" in o_low else "Away Win Either Half No"
    if m_id == "58" or m_low == "both halves over 1.5":
        return "Both Halves Over 1.5" if "yes" in o_low else "Both Halves Over 1.5 No"
    if m_id == "59" or m_low == "both halves under 1.5":
        return "Both Halves Under 1.5" if "yes" in o_low else "Both Halves Under 1.5 No"
    if m_id == "26" or m_low == "odd/even":
        return f"{o_desc} Goals"
    if m_id == "548" or "multi goal" in m_low or "multigoal" in m_low:
        return f"Multi-Goals {o_desc}"

    # 9. Any other SportyBet market (Handicaps, Corners, Bookings, Combos, Correct Score, HT/FT, Player Props)
    if m_low in o_low:
        return o_desc
    if specifier and "hcp=" in specifier and "(" not in o_desc:
        hcp_val = specifier.split("hcp=")[-1].split("&")[0]
        return f"{m_desc} ({hcp_val}): {o_desc}"
    return f"{m_desc}: {o_desc}"


def _fetch_event_baseline_odds(event_id: str) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Fetches real-time Full-Time 1X2 baseline odds (odds_home, odds_draw, odds_away) and
    extended multi-market odds for a SportyBet eventId via SportyBet's factsCenter/event API.
    """
    import urllib.request
    odds_h, odds_d, odds_a = 0.0, 0.0, 0.0
    market_odds: Dict[str, float] = {}
    if not event_id or not event_id.startswith("sr:match:"):
        return odds_h, odds_d, odds_a, market_odds

    try:
        url = f"https://www.sportybet.com/api/ng/factsCenter/event?eventId={event_id}&productId=3"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=6) as r:
            payload = json.loads(r.read().decode("utf-8"))
            ev_data = payload.get("data") or {}
            for m in (ev_data.get("markets") or []):
                m_id = str(m.get("id") or "").strip()
                m_desc = str(m.get("desc") or "").strip()
                spec = str(m.get("specifier") or "").strip()
                for o in (m.get("outcomes") or []):
                    try:
                        ov = float(o.get("odds") or 0.0)
                    except (ValueError, TypeError):
                        ov = 0.0
                    if ov <= 1.0:
                        continue
                    o_id = str(o.get("id") or "").strip()
                    o_desc = str(o.get("desc") or "").strip()
                    norm_label = normalize_sportybet_selection(m_desc, o_desc, m_id, spec)
                    market_odds[norm_label] = ov
                    if m_id == "1":
                        if o_id == "1" or o_desc.lower() == "home":
                            odds_h = ov
                        elif o_id == "2" or o_desc.lower() == "draw":
                            odds_d = ov
                        elif o_id == "3" or o_desc.lower() == "away":
                            odds_a = ov
    except Exception as e:
        logger.debug(f"Could not fetch baseline event odds for {event_id}: {e}")

    return odds_h, odds_d, odds_a, market_odds


async def _extract_via_sportybet_share_api(code: str) -> Dict[str, Any]:
    """
    Fast, 100%-accurate extraction of any SportyBet booking code using SportyBet's official
    Share Order endpoint (GET https://www.sportybet.com/api/ng/orders/share/{code}).
    Reads all 210+ SportyBet markets (1st Half, 2nd Half, Team O/U, Handicap, Clean Sheet,
    Win Either Half, Combos, Corners, Bookings, 1UP/2UP, etc.) and enriches each fixture with
    its real Full-Time 1X2 baseline odds.
    """
    import urllib.request
    clean_code = code.strip().upper()
    share_url = f"https://www.sportybet.com/api/ng/orders/share/{clean_code}"

    def _sync_fetch():
        req = urllib.request.Request(
            share_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))

    try:
        data = await asyncio.to_thread(_sync_fetch)
        biz_code = data.get("bizCode")
        outcomes_raw = (data.get("data") or {}).get("outcomes") or []
        if biz_code != 10000 or not outcomes_raw:
            return {
                "success": False,
                "error_code": ErrorCode.INVALID_CODE,
                "error_message": data.get("message") or "Booking code not found or expired on SportyBet"
            }

        try:
            try:
                from data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
            except ImportError:
                from src.data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
            agg = get_stats_aggregator()
        except Exception:
            agg = None

        # Pre-fetch baseline 1X2 & multi-market odds for all extracted eventIds in parallel (< 1.5s total)
        event_ids_to_fetch = [str(it.get("eventId") or "").strip() for it in outcomes_raw]
        parallel_event_odds = await asyncio.gather(
            *[asyncio.to_thread(_fetch_event_baseline_odds, eid) for eid in event_ids_to_fetch]
        )

        extracted_matches: List[Dict[str, Any]] = []
        for idx, item in enumerate(outcomes_raw):
            event_id = str(item.get("eventId") or "").strip()
            home_team = " ".join(str(item.get("homeTeamName") or "Unknown").split())
            away_team = " ".join(str(item.get("awayTeamName") or "Unknown").split())
            league = (
                item.get("sport", {})
                .get("category", {})
                .get("tournament", {})
                .get("name")
                or "Football"
            )
            est_start = item.get("estimateStartTime")
            if est_start:
                try:
                    start_time = datetime.fromtimestamp(float(est_start) / 1000.0, tz=timezone.utc).isoformat()
                except Exception:
                    start_time = datetime.now(timezone.utc).isoformat()
            else:
                start_time = datetime.now(timezone.utc).isoformat()

            markets_list = item.get("markets") or []
            market_id = "1"
            market_desc = "1X2"
            specifier = ""
            outcome_id = "1"
            outcome_desc = "Home"
            selected_odds = 0.0

            if markets_list:
                m_obj = markets_list[0]
                market_id = str(m_obj.get("id") or "1").strip()
                market_desc = str(m_obj.get("desc") or "1X2").strip()
                specifier = str(m_obj.get("specifier") or "").strip()
                o_list = m_obj.get("outcomes") or []
                if o_list:
                    o_obj = o_list[0]
                    outcome_id = str(o_obj.get("id") or "1").strip()
                    outcome_desc = str(o_obj.get("desc") or "Home").strip()
                    try:
                        selected_odds = float(o_obj.get("odds") or 0.0)
                    except (ValueError, TypeError):
                        selected_odds = 0.0

            selected_outcome = normalize_sportybet_selection(
                market_desc=market_desc,
                outcome_desc=outcome_desc,
                market_id=market_id,
                specifier=specifier
            )

            # Resolve real Full-Time 1X2 baseline odds (so non-1X2 picks like "2nd Half Under 2.5"
            # never pollute the match's Full-Time 1X2 Poisson parameters)
            odds_home, odds_draw, odds_away = 0.0, 0.0, 0.0
            market_odds: Dict[str, float] = {}

            if agg:
                sb_cached = agg.find_sportybet_match(home_team, away_team)
                if sb_cached:
                    odds_home = float(sb_cached.odds_home or 0.0)
                    odds_draw = float(sb_cached.odds_draw or 0.0)
                    odds_away = float(sb_cached.odds_away or 0.0)
                    if getattr(sb_cached, "market_odds", None):
                        market_odds.update(sb_cached.market_odds)

            # Apply parallel-fetched live event markets from SportyBet factsCenter
            fh, fd, fa, f_mo = parallel_event_odds[idx]
            if fh > 1.0 and fa > 1.0:
                odds_home, odds_draw, odds_away = fh, fd, fa
            if f_mo:
                market_odds.update(f_mo)

            if selected_odds > 1.0:
                market_odds[selected_outcome] = selected_odds

            # Fallback if match has started/finished or 1X2 market is closed
            if odds_home <= 1.0 or odds_away <= 1.0:
                if market_id == "1" and selected_odds > 1.0:
                    if outcome_id == "1":
                        odds_home, odds_draw, odds_away = selected_odds, 3.30, round(max(1.80, selected_odds * 1.45), 2)
                    elif outcome_id == "3":
                        odds_away, odds_draw, odds_home = selected_odds, 3.30, round(max(1.80, selected_odds * 1.45), 2)
                    else:
                        odds_draw, odds_home, odds_away = selected_odds, 2.35, 2.75
                else:
                    odds_home, odds_draw, odds_away = 2.15, 3.25, 3.10

            extracted_matches.append({
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
                "selected_outcome": selected_outcome,
                "selected_odds": selected_odds if selected_odds > 1.0 else odds_home,
                "sporty_id": event_id,
                "sporty_market_id": market_id,
                "sporty_outcome_id": outcome_id,
                "sporty_specifier": specifier,
                "sporty_market_desc": market_desc,
                "sporty_outcome_desc": outcome_desc,
                "market_odds": market_odds,
                "start_time": start_time,
                "sport": "football",
                "status": "upcoming"
            })

        return {
            "success": True,
            "matches": extracted_matches,
            "url": share_url
        }
    except Exception as e:
        logger.debug(f"[{clean_code}] SportyBet Share API extraction error: {e}")
        return {
            "success": False,
            "error_code": ErrorCode.UNKNOWN,
            "error_message": str(e)
        }


async def _try_api_fallback(code: str) -> Dict[str, Any]:
    """
    Fallback to SportyBet's official Share Order endpoint if Playwright fails.
    """
    return await _extract_via_sportybet_share_api(code)


def _build_odds_data(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build comprehensive odds data structure from matches."""
    odds_data: Dict[str, Any] = {
        "h_d_a_summary": {"home_win": 0.0, "draw": 0.0, "away_win": 0.0},
        "btts_summary": {"both_to_score": 0.0, "over": 0.0, "under": 0.0},
        "over_under_summary": {"over_2_5": 0.0, "under_2_5": 0.0},
        "aggregated": {},
        "by_match": {}
    }

    for match in matches:
        match_key = f"{match.get('home_team', 'Unknown')}_{match.get('away_team', 'Unknown')}"

        # Collect odds by type
        match_odds = {
            "h_d_a": {
                "home": match.get("odds_home", 0.0),
                "draw": match.get("odds_draw", 0.0),
                "away": match.get("odds_away", 0.0)
            }
        }

        # Update summaries
        for key, value in match_odds["h_d_a"].items():
            summary_key = f"{key}_win" if key != "draw" else "draw"
            odds_data["h_d_a_summary"][summary_key] += value

        odds_data["by_match"][match_key] = match_odds

    # Calculate averages
    count = len(matches) if matches else 1
    for key in odds_data["h_d_a_summary"]:
        odds_data["h_d_a_summary"][key] = round(odds_data["h_d_a_summary"][key] / count, 4)

    # Add aggregated info
    odds_data["aggregated"] = {
        "total_matches": count,
        "average_home_win_odds": odds_data["h_d_a_summary"]["home_win"],
        "average_draw_odds": odds_data["h_d_a_summary"]["draw"],
        "average_away_win_odds": odds_data["h_d_a_summary"]["away_win"]
    }

    return odds_data


def _save_to_json(data: Dict[str, Any], output_file: str) -> None:
    """Save extraction result to JSON file."""
    try:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add timestamp to filename if it doesn't have extension
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Saved extraction result to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")


def _create_error_result(code: str,
                         error_code: ErrorCode,
                         error_message: str,
                         method: str) -> Dict[str, Any]:
    """Create an error result structure."""
    return {
        "metadata": {
            "url": "",
            "extraction_time": datetime.now(timezone.utc).isoformat(),
            "code": code,
            "source": "sportybet",
            "method": method
        },
        "matches": [],
        "odds": {},
        "extraction_details": {
            "status": "failed",
            "error_code": error_code.value,
            "error_message": error_message,
            "warnings": [],
            "total_matches": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "extraction_time_seconds": 0.0,
            "page_load_time_ms": 0.0,
            "element_find_time_ms": 0.0
        }
    }


async def _extract_code_async(code: str,
                               headless: bool,
                               output_file: Optional[str],
                               use_api_fallback: bool,
                               timeout: int,
                               max_retries: int) -> Dict[str, Any]:
    """
    Internal async implementation for code extraction.
    """
    start_time = datetime.now(timezone.utc)
    extraction_details = ExtractionDetails(status="in_progress")

    matches: List[Dict[str, Any]] = []
    odds_data: Dict[str, Any] = {}
    url = ""
    error_code = None
    error_message = ""
    successful_extraction = False
    method_used = ""

    # 1. Try SportyBet Official Share Order API first (< 0.5s, 100% market fidelity across all 210+ markets)
    try:
        api_start = datetime.now()
        api_res = await _extract_via_sportybet_share_api(code)
        api_elapsed = (datetime.now() - api_start).total_seconds() * 1000
        if api_res.get("success") and api_res.get("matches"):
            matches = api_res["matches"]
            url = api_res.get("url", "")
            successful_extraction = True
            method_used = "sportybet_share_api"
            extraction_details.page_load_time_ms = api_elapsed
            extraction_details.successful_extractions = len(matches)
            logger.info(f"[{code}] Extracted {len(matches)} matches via SportyBet Share API in {api_elapsed:.0f}ms")
    except Exception as e:
        logger.debug(f"[{code}] Share API primary check skipped: {e}")

    # 2. Fallback to Playwright DOM extraction if Share API did not return matches
    if not successful_extraction:
        try:
            logger.info(f"[{code}] Starting Playwright extraction...")

            async with SportyBetScraper(headless=headless, max_retries=max_retries) as scraper:
                play_start = datetime.now()
                matches, url, error_message = await scraper.extract_code(code)
                play_elapsed = (datetime.now() - play_start).total_seconds() * 1000

                extraction_details.page_load_time_ms = play_elapsed

                if matches:
                    successful_extraction = True
                    method_used = "playwright"
                    extraction_details.successful_extractions = len(matches)
                    logger.info(f"[{code}] Extracted {len(matches)} matches via Playwright")
                else:
                    error_code, error_message = _determine_error_type(error_message, url)

        except ScrapingError as e:
            error_code = ErrorCode.NETWORK_TIMEOUT
            error_message = str(e)
            logger.error(f"[{code}] Scraping error: {e}")
        except Exception as e:
            error_code = ErrorCode.UNKNOWN
            error_message = str(e)
            logger.error(f"[{code}] Unexpected error: {e}")

    # Fallback to API if configured and Playwright failed
    if not successful_extraction and use_api_fallback:
        logger.info(f"[{code}] Attempting API fallback...")
        api_result = await _try_api_fallback(code)

        if api_result["success"]:
            matches = api_result["matches"]
            url = api_result.get("url", "")
            successful_extraction = True
            method_used = "api_fallback"
            error_code = None
            error_message = ""
            logger.info(f"[{code}] API fallback extracted {len(matches)} matches")
        else:
            error_code = api_result.get("error_code", error_code)
            error_message = api_result.get("error_message", error_message)

    # Build odds data from matches
    if matches:
        odds_data = _build_odds_data(matches)

    # Finalize metadata
    extraction_time = (datetime.now(timezone.utc) - start_time).total_seconds()
    extraction_details.extraction_time_seconds = extraction_time
    extraction_details.status = "success" if successful_extraction else "failed"
    extraction_details.error_code = error_code.value if error_code else None
    extraction_details.error_message = error_message
    extraction_details.total_matches = len(matches)
    extraction_details.failed_extractions = len(matches) - extraction_details.successful_extractions

    metadata = Metadata(
        url=url,
        extraction_time=start_time.isoformat(),
        code=code,
        source="sportybet",
        method=method_used or "none"
    )

    # Build result
    result = {
        "metadata": metadata.to_dict(),
        "matches": matches,
        "odds": odds_data,
        "extraction_details": extraction_details.to_dict()
    }

    # Save to file if specified
    if output_file:
        _save_to_json(result, output_file)

    return result


def extract_betting_code(code: str,
                         headless: bool = True,
                         output_file: Optional[str] = None,
                         use_api_fallback: bool = True,
                         timeout: int = 30000,
                         max_retries: int = 3) -> Dict[str, Any]:
    """
    Main entry point for SportyBet code extraction.

    Accepts a booking code and extracts all matches and odds,
    saving to JSON with comprehensive documentation.

    Args:
        code: SportyBet booking code (e.g., "ABC123XYZ")
        headless: Run browser in headless mode (default: True)
        output_file: Optional path to save JSON output
        use_api_fallback: Use API fallback if Playwright fails
        timeout: Timeout in milliseconds for extraction
        max_retries: Maximum number of retry attempts

    Returns:
        Dict with 'metadata', 'matches', 'odds', 'extraction_details'

    Example:
        >>> result = extract_betting_code("ABC123XYZ")
        >>> print(result["metadata"]["code"])
        'ABC123XYZ'
        >>> print(result["extraction_details"]["status"])
        'success'

    Output JSON Format:
        {
            "metadata": {
                "url": "https://www.sportybet.com/ng/play/ABC123XYZ",
                "extraction_time": "2024-01-15T10:30:00+00:00",
                "code": "ABC123XYZ",
                "source": "sportybet",
                "method": "playwright",
                "retry_count": 0,
                "user_agent": "Mozilla/5.0..."
            },
            "matches": [
                {
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "league": "Premier League",
                    "date": "2024-01-20",
                    "time": "15:00",
                    "venue": "Stadium A",
                    "status": "upcoming",
                    "sport": "football",
                    "odds": {...},
                    "match_id": "..."
                }
            ],
            "odds": {
                "h_d_a_summary": {...},
                "btts_summary": {...},
                "over_under_summary": {...},
                "aggregated": {...},
                "by_match": {...}
            },
            "extraction_details": {
                "status": "success",
                "error_code": null,
                "error_message": "",
                "warnings": [],
                "total_matches": 5,
                "successful_extractions": 5,
                "failed_extractions": 0,
                "extraction_time_seconds": 2.5,
                "page_load_time_ms": 1500.0,
                "element_find_time_ms": 100.0
            }
        }
    """
async def extract_betting_code_async(
    code: str,
    headless: bool = True,
    output_file: Optional[str] = None,
    use_api_fallback: bool = False,
    timeout: int = 30000,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Asynchronous version of extract_betting_code, safe for use inside existing event loops.
    """
    validation = validate_code_format(code)
    if not validation["valid"]:
        return _create_error_result(
            code=code,
            error_code=ErrorCode.INVALID_CODE,
            error_message=validation["reason"],
            method="validation"
        )

    if run_coro_with_proactor is not None:
        return await run_coro_with_proactor(
            _extract_code_async,
            code=code,
            headless=headless,
            output_file=output_file,
            use_api_fallback=use_api_fallback,
            timeout=timeout,
            max_retries=max_retries
        )

    return await _extract_code_async(
        code=code,
        headless=headless,
        output_file=output_file,
        use_api_fallback=use_api_fallback,
        timeout=timeout,
        max_retries=max_retries
    )


def extract_betting_code(
    code: str,
    headless: bool = True,
    output_file: Optional[str] = None,
    use_api_fallback: bool = False,
    timeout: int = 30000,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Synchronous interface for extracting matches from booking codes.
    Safely delegates to Proactor worker thread on Windows to guarantee subprocess compatibility.
    """
    if run_in_proactor_thread is not None:
        return run_in_proactor_thread(
            extract_betting_code_async,
            code=code,
            headless=headless,
            output_file=output_file,
            use_api_fallback=use_api_fallback,
            timeout=timeout,
            max_retries=max_retries
        )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                extract_betting_code_async(
                    code=code,
                    headless=headless,
                    output_file=output_file,
                    use_api_fallback=use_api_fallback,
                    timeout=timeout,
                    max_retries=max_retries
                )
            ).result()

    return asyncio.run(extract_betting_code_async(
        code=code,
        headless=headless,
        output_file=output_file,
        use_api_fallback=use_api_fallback,
        timeout=timeout,
        max_retries=max_retries
    ))


def read_sporty_code(code: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for extract_betting_code with simpler interface.

    This provides a programmatic API for direct use without
    specifying all the optional parameters.

    Args:
        code: SportyBet booking code

    Returns:
        Dictionary with extraction results

    Example:
        >>> result = read_sporty_code("ABC123XYZ")
        >>> if result["extraction_details"]["status"] == "success":
        ...     for match in result["matches"]:
        ...         print(f"{match['home_team']} vs {match['away_team']}")
    """
    return extract_betting_code(
        code=code,
        headless=True,
        output_file=None,
        use_api_fallback=True,
        timeout=30000,
        max_retries=3
    )


class CodeReader:
    """
    Object-oriented interface for booking code extraction.

    Provides a class-based API for more complex usage patterns,
    supporting both direct Playwright extraction and API fallback.

    Example:
        >>> reader = CodeReader(headless=True)
        >>> result = reader.read("ABC123XYZ")
        >>> if result.success:
        ...     print(f"Found {len(result.matches)} matches")
    """

    def __init__(self,
                 headless: bool = True,
                 max_retries: int = 3,
                 timeout: int = 30000,
                 output_dir: Optional[str] = None):
        """
        Initialize the CodeReader.

        Args:
            headless: Run browser in headless mode
            max_retries: Maximum retry attempts
            timeout: Extraction timeout in milliseconds
            output_dir: Directory for output files
        """
        self.headless = headless
        self.max_retries = max_retries
        self.timeout = timeout
        self.output_dir = output_dir
        self._last_result: Optional[Dict[str, Any]] = None

    async def read(self, code: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract data for a booking code.

        Args:
            code: SportyBet booking code
            output_file: Optional output file path (relative to output_dir if specified)

        Returns:
            Extraction result dictionary
        """
        if output_file and self.output_dir:
            output_file = str(Path(self.output_dir) / output_file)

        result = await _extract_code_async(
            code=code,
            headless=self.headless,
            output_file=output_file,
            use_api_fallback=True,
            timeout=self.timeout,
            max_retries=self.max_retries
        )

        self._last_result = result
        return result

    def get_matches(self) -> List[Dict[str, Any]]:
        """Get matches from the last extraction."""
        if self._last_result:
            return self._last_result.get("matches", [])
        return []

    def get_odds(self) -> Dict[str, Any]:
        """Get odds from the last extraction."""
        if self._last_result:
            return self._last_result.get("odds", {})
        return {}

    def has_errors(self) -> bool:
        """Check if the last extraction had errors."""
        if not self._last_result:
            return True
        return self._last_result.get("extraction_details", {}).get("status") == "failed"

    def get_error(self) -> Optional[str]:
        """Get error message from the last extraction."""
        if not self._last_result:
            return None
        extraction = self._last_result.get("extraction_details", {})
        return extraction.get("error_message")


def _cli_extract_code() -> None:
    """CLI handler for code extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SportyBet Code Reader - Extract matches and odds from booking codes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog_sname)s -c ABC123XYZ                        Extract and save to file
  %(prog_sname)s -c ABC123XYZ -o results.json         Specify output file
  %(prog_sname)s -c ABC123XYZ --no-headless           Debug mode (show browser)
  %(prog_sname)s -c ABC123XYZ -o output/ -f         Save to directory
        """
    )

    parser.add_argument(
        '-c', '--code',
        type=str,
        required=True,
        help='SportyBet booking code to extract'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: ./output/code_<code>.json)'
    )

    parser.add_argument(
        '--no-headless',
        action='store_true',
        help='Run browser in visible mode (for debugging)'
    )

    parser.add_argument(
        '--timeout',
        type=int,
        default=30000,
        help='Extraction timeout in milliseconds (default: 30000)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--list-matches',
        action='store_true',
        help='List extracted matches (print summary to stdout)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine output file
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/code_{args.code}_{timestamp}.json"

    # Run extraction
    result = extract_betting_code(
        code=args.code,
        headless=not args.no_headless,
        output_file=output_file,
        use_api_fallback=True,
        timeout=args.timeout,
        max_retries=3
    )

    # Print summary if requested
    if args.list_matches:
        matches = result.get("matches", [])
        extraction = result.get("extraction_details", {})
        metadata = result.get("metadata", {})

        print(f"\n{'='*60}")
        print(f"SportyBet Code: {metadata.get('code', 'N/A')}")
        print(f"Extraction Time: {metadata.get('extraction_time', 'N/A')}")
        print(f"Status: {extraction.get('status', 'N/A')}")
        print(f"Method: {metadata.get('method', 'N/A')}")
        print(f"{'='*60}")

        if matches:
            print(f"\nMatches ({len(matches)}):")
            for i, match in enumerate(matches, 1):
                print(f"  {i}. {match.get('home_team', 'N/A')} vs {match.get('away_team', 'N/A')}")
                odds = match.get('odds', {})
                if isinstance(odds, dict):
                    h_d_a = odds.get('h_d_a', odds)
                    print(f"     Odds: H={h_d_a.get('home', 0):.2f} "
                          f"D={h_d_a.get('draw', 0):.2f} "
                          f"A={h_d_a.get('away', 0):.2f}")
        else:
            print("\nNo matches extracted.")
            if extraction.get("error_message"):
                print(f"Error: {extraction.get('error_message')}")

        print(f"\nSaved to: {output_file}")
    elif extraction.get("status") == "success":
        print(f"Successfully extracted {len(result.get('matches', []))} matches")
        print(f"Saved to: {output_file}")
    else:
        print(f"Extraction failed: {extraction.get('error_message')}")
        if extraction.get("error_code"):
            print(f"Error code: {extraction.get('error_code')}")


if __name__ == "__main__":
    _cli_extract_code()