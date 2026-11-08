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


async def _try_api_fallback(code: str) -> Dict[str, Any]:
    """
    Pure web scraping mode: external API fallback is disabled by design.
    """
    logger.debug("Pure web scraping architecture: external odds API fallback disabled")
    return {
        "success": False,
        "error_code": ErrorCode.UNKNOWN,
        "error_message": "Pure web scraping architecture active - external API fallback disabled"
    }


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

    # Try Playwright extraction first
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