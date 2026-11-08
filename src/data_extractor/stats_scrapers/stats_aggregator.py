"""
Statistical Aggregator & Multi-Site Intelligence Hub
Unites predictions, mathematical probabilities, xG approximations,
and form guides scraped from Forebet and PredictZ.
"""

import asyncio
import difflib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

from .forebet_scraper import ForebetScraper, ForebetPrediction
from .predictz_scraper import PredictZScraper, PredictZPrediction
from .matchoutlook_scraper import MatchOutlookScraper, MatchOutlookPrediction
from .scoregpt_scraper import ScoreGPTScraper, ScoreGPTPrediction
from .betensured_scraper import BetensuredScraper, BetensuredPrediction
from .sportybet_odds_scraper import SportyBetOddsScraper, SportyBetFixtureOdds

try:
    from utils.proactor_helper import run_coro_with_proactor
except ImportError:
    try:
        from src.utils.proactor_helper import run_coro_with_proactor
    except ImportError:
        try:
            from ...utils.proactor_helper import run_coro_with_proactor
        except ImportError:
            run_coro_with_proactor = None

logger = logging.getLogger(__name__)


@dataclass
class AggregatedStats:
    """Consolidated statistical intelligence for a single fixture"""
    match_str: str
    home_team: str
    away_team: str
    
    # Mathematical Forebet predictions
    has_forebet: bool = False
    forebet_p_home: float = 0.33
    forebet_p_draw: float = 0.33
    forebet_p_away: float = 0.34
    forebet_score: str = ""
    forebet_avg_goals: float = 2.5
    
    # PredictZ intelligence
    has_predictz: bool = False
    predictz_outcome: str = ""
    predictz_score: str = ""
    home_form: List[str] = field(default_factory=list)
    away_form: List[str] = field(default_factory=list)

    # MatchOutlook specialized tips (totals, double chance, etc.)
    has_matchoutlook: bool = False
    matchoutlook_best_bet: str = ""
    matchoutlook_desc: str = ""

    # ScoreGPT multi-model AI consensus
    has_scoregpt: bool = False
    scoregpt_pick: str = ""
    scoregpt_score: str = ""
    scoregpt_consensus: str = ""
    scoregpt_panel: str = ""

    # Betensured algorithmic predictions & tips
    has_betensured: bool = False
    betensured_tip: str = ""
    betensured_score: str = ""
    betensured_p_home: int = 0
    betensured_p_draw: int = 0
    betensured_p_away: int = 0
    
    # Derived Statistical Metrics
    est_home_xg: float = 1.5
    est_away_xg: float = 1.2
    est_home_xga: float = 1.2
    est_away_xga: float = 1.5
    home_form_factor: float = 0.5  # 0 to 1 based on W/D/L
    away_form_factor: float = 0.5
    consensus_home_prob: float = 0.33
    consensus_draw_prob: float = 0.33
    consensus_away_prob: float = 0.34

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match_str,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "has_forebet": self.has_forebet,
            "forebet_prob": {
                "home": round(self.forebet_p_home, 4),
                "draw": round(self.forebet_p_draw, 4),
                "away": round(self.forebet_p_away, 4)
            },
            "forebet_score": self.forebet_score,
            "forebet_avg_goals": round(self.forebet_avg_goals, 2),
            "has_predictz": self.has_predictz,
            "predictz_score": self.predictz_score,
            "predictz_outcome": self.predictz_outcome,
            "home_form": self.home_form,
            "away_form": self.away_form,
            "has_matchoutlook": self.has_matchoutlook,
            "matchoutlook_best_bet": self.matchoutlook_best_bet,
            "matchoutlook_desc": self.matchoutlook_desc,
            "has_scoregpt": self.has_scoregpt,
            "scoregpt_pick": self.scoregpt_pick,
            "scoregpt_score": self.scoregpt_score,
            "scoregpt_consensus": self.scoregpt_consensus,
            "scoregpt_panel": self.scoregpt_panel,
            "has_betensured": self.has_betensured,
            "betensured_tip": self.betensured_tip,
            "betensured_score": self.betensured_score,
            "betensured_prob": {
                "home": self.betensured_p_home,
                "draw": self.betensured_p_draw,
                "away": self.betensured_p_away
            },
            "est_home_xg": round(self.est_home_xg, 2),
            "est_away_xg": round(self.est_away_xg, 2),
            "est_home_xga": round(self.est_home_xga, 2),
            "est_away_xga": round(self.est_away_xga, 2),
            "consensus_prob": {
                "home": round(self.consensus_home_prob, 4),
                "draw": round(self.consensus_draw_prob, 4),
                "away": round(self.consensus_away_prob, 4)
            }
        }


class StatsAggregator:
    """
    Central hub for fetching, caching, and mapping external statistics
    to SportyBet matches.
    """

    def __init__(self, cache_ttl_seconds: int = 1800):
        self.cache_ttl = cache_ttl_seconds
        self._sportybet_cache: List[SportyBetFixtureOdds] = []
        self._forebet_cache: List[ForebetPrediction] = []
        self._predictz_cache: List[PredictZPrediction] = []
        self._matchoutlook_cache: List[MatchOutlookPrediction] = []
        self._scoregpt_cache: List[ScoreGPTPrediction] = []
        self._betensured_cache: List[BetensuredPrediction] = []
        self._last_refresh: float = 0.0
        self._lock = asyncio.Lock()
        self._catalog_cache: List[Dict[str, Any]] = []
        self._load_cache_from_disk()

    def _disk_cache_path(self):
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent.parent.parent / "data" / "scraped_cache.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _load_cache_from_disk(self):
        try:
            p = self._disk_cache_path()
            if p.exists():
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                self._sportybet_cache = [SportyBetFixtureOdds(**m) for m in data.get("sportybet", [])]
                self._forebet_cache = [ForebetPrediction(**m) for m in data.get("forebet", [])]
                self._predictz_cache = [PredictZPrediction(**m) for m in data.get("predictz", [])]
                self._matchoutlook_cache = [MatchOutlookPrediction(**m) for m in data.get("matchoutlook", [])]
                self._scoregpt_cache = [ScoreGPTPrediction(**m) for m in data.get("scoregpt", [])]
                self._betensured_cache = [BetensuredPrediction(**m) for m in data.get("betensured", [])]
                self._last_refresh = data.get("timestamp", time.time())
        except Exception as e:
            logger.debug(f"Failed to load disk cache: {e}")

    def _save_cache_to_disk(self):
        try:
            p = self._disk_cache_path()
            import json
            data = {
                "timestamp": self._last_refresh,
                "sportybet": [m.to_dict() for m in self._sportybet_cache],
                "forebet": [m.to_dict() for m in self._forebet_cache],
                "predictz": [m.to_dict() for m in self._predictz_cache],
                "matchoutlook": [m.to_dict() for m in self._matchoutlook_cache],
                "scoregpt": [m.to_dict() for m in self._scoregpt_cache],
                "betensured": [m.to_dict() for m in self._betensured_cache]
            }
            p.write_text(json.dumps(data), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to save disk cache: {e}")

    def _normalize_name(self, name: str) -> str:
        """Normalize team names for fast fuzzy matching"""
        name = name.lower().strip()
        replacements = [
            ("fc", ""), ("afc", ""), ("cf", ""), ("sc", ""), ("fk", ""),
            ("manchester", "man"), ("united", "utd"),
            ("hotspur", ""), ("athletic", ""), ("club", ""),
            ("paris saint germain", "psg"), ("paris sg", "psg"),
            ("borussia dortmund", "dortmund"), ("bvb", "dortmund"),
            ("bayern munich", "bayern"), ("bayern munchen", "bayern"),
            ("internazionale", "inter"), ("inter milan", "inter"),
            ("sporting lisbon", "sporting"), ("sporting cp", "sporting"),
            ("real betis", "betis"), ("vfb stuttgart", "stuttgart")
        ]
        for old, new in replacements:
            name = re.sub(rf'\b{old}\b', new, name)
        return " ".join(name.split())

    def _match_similarity(self, team_a: str, team_b: str) -> float:
        """Compute string similarity ratio between two team names"""
        norm_a = self._normalize_name(team_a)
        norm_b = self._normalize_name(team_b)
        if norm_a == norm_b or norm_a in norm_b or norm_b in norm_a:
            return 0.95
        tokens_a = set(norm_a.split())
        tokens_b = set(norm_b.split())
        if tokens_a and tokens_b and (tokens_a.issubset(tokens_b) or tokens_b.issubset(tokens_a)):
            return 0.90
        overlap = len(tokens_a.intersection(tokens_b))
        if overlap > 0 and len(tokens_a) > 0 and len(tokens_b) > 0:
            jaccard = overlap / len(tokens_a.union(tokens_b))
            if jaccard >= 0.5:
                return 0.85
        if not norm_a or not norm_b or (norm_a[0] != norm_b[0] and overlap == 0):
            return 0.0
        return difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    async def refresh_if_needed(self, force: bool = False):
        """Fetch fresh predictions if cache expired"""
        now = time.time()
        if not force and (now - self._last_refresh < self.cache_ttl) and (self._forebet_cache or self._predictz_cache):
            return

        async with self._lock:
            if not force and (now - self._last_refresh < self.cache_ttl) and (self._forebet_cache or self._predictz_cache or self._sportybet_cache):
                return

            logger.info("Refreshing live bookmaker odds and external statistics from SportyBet, Forebet, PredictZ & MatchOutlook...")

            # 1. Scrape SportyBet (Official Ground Truth Odds)
            try:
                async def _run_sb():
                    async with SportyBetOddsScraper(headless=True) as sb_scraper:
                        return await sb_scraper.scrape_live_odds()

                if run_coro_with_proactor is not None:
                    self._sportybet_cache = await run_coro_with_proactor(_run_sb)
                else:
                    self._sportybet_cache = await _run_sb()
            except Exception as e:
                logger.warning(f"SportyBet DOM scrape failed ({e}), falling back to direct API...")
                try:
                    self._sportybet_cache = SportyBetOddsScraper()._scrape_via_api()
                except Exception as e2:
                    logger.error(f"Failed to refresh SportyBet: {e2}")

            # Fast supplement from SportyBet API if needed
            if len(self._sportybet_cache) < 20:
                try:
                    api_sb = SportyBetOddsScraper()._scrape_via_api()
                    existing_keys = {f"{f.home_team.lower()}_vs_{f.away_team.lower()}" for f in self._sportybet_cache}
                    for af in api_sb:
                        k = f"{af.home_team.lower()}_vs_{af.away_team.lower()}"
                        if k not in existing_keys:
                            self._sportybet_cache.append(af)
                            existing_keys.add(k)
                except Exception:
                    pass

            async def _run_fb():
                async with ForebetScraper(headless=True) as fb_scraper:
                    return await fb_scraper.scrape_predictions()

            async def _run_pz():
                async with PredictZScraper(headless=True) as pz_scraper:
                    return await pz_scraper.scrape_all_leagues()

            async def _run_mo():
                async with MatchOutlookScraper(headless=True) as mo_scraper:
                    return await mo_scraper.scrape_predictions()

            try:
                # Scrape Forebet
                if run_coro_with_proactor is not None:
                    self._forebet_cache = await run_coro_with_proactor(_run_fb)
                else:
                    self._forebet_cache = await _run_fb()
            except Exception as e:
                logger.error(f"Failed to refresh Forebet: {e}")

            try:
                # Scrape PredictZ
                if run_coro_with_proactor is not None:
                    self._predictz_cache = await run_coro_with_proactor(_run_pz)
                else:
                    self._predictz_cache = await _run_pz()
            except Exception as e:
                logger.error(f"Failed to refresh PredictZ: {e}")

            try:
                # Scrape MatchOutlook
                if run_coro_with_proactor is not None:
                    self._matchoutlook_cache = await run_coro_with_proactor(_run_mo)
                else:
                    self._matchoutlook_cache = await _run_mo()
            except Exception as e:
                logger.error(f"Failed to refresh MatchOutlook: {e}")

            try:
                # Scrape ScoreGPT (Multi-Model AI)
                async with ScoreGPTScraper() as sg_scraper:
                    self._scoregpt_cache = await sg_scraper.scrape_predictions()
            except Exception as e:
                logger.error(f"Failed to refresh ScoreGPT: {e}")

            try:
                # Scrape Betensured
                async with BetensuredScraper() as be_scraper:
                    self._betensured_cache = await be_scraper.scrape_predictions()
            except Exception as e:
                logger.error(f"Failed to refresh Betensured: {e}")

            self._last_refresh = time.time()
            self._catalog_cache = []
            self._save_cache_to_disk()
            logger.info(f"Statistics refreshed: {len(self._sportybet_cache)} SportyBet, {len(self._forebet_cache)} Forebet, {len(self._predictz_cache)} PredictZ, {len(self._matchoutlook_cache)} MatchOutlook, {len(self._scoregpt_cache)} ScoreGPT, {len(self._betensured_cache)} Betensured")

    def find_sportybet_match(self, home: str, away: str) -> Optional[SportyBetFixtureOdds]:
        """Find SportyBet match with real live bookmaker odds"""
        best_match = None
        best_score = 0.0

        for sb in self._sportybet_cache:
            sim_h = self._match_similarity(home, sb.home_team)
            sim_a = self._match_similarity(away, sb.away_team)
            score = (sim_h + sim_a) / 2.0

            if score > 0.55 and score > best_score:
                best_score = score
                best_match = sb

        return best_match

    def find_forebet_match(self, home: str, away: str) -> Optional[ForebetPrediction]:
        """Find Forebet prediction matching home & away team"""
        best_match = None
        best_score = 0.0

        for fb in self._forebet_cache:
            sim_h = self._match_similarity(home, fb.home_team)
            sim_a = self._match_similarity(away, fb.away_team)
            score = (sim_h + sim_a) / 2.0

            if score > 0.55 and score > best_score:
                best_score = score
                best_match = fb

        return best_match

    def find_predictz_match(self, home: str, away: str) -> Optional[PredictZPrediction]:
        """Find PredictZ prediction matching home & away team"""
        best_match = None
        best_score = 0.0

        for pz in self._predictz_cache:
            sim_h = self._match_similarity(home, pz.home_team)
            sim_a = self._match_similarity(away, pz.away_team)
            score = (sim_h + sim_a) / 2.0

            if score > 0.55 and score > best_score:
                best_score = score
                best_match = pz

        return best_match

    def find_matchoutlook_match(self, home: str, away: str) -> Optional[MatchOutlookPrediction]:
        """Find MatchOutlook prediction matching home & away team"""
        best_match = None
        best_score = 0.0

        for mo in self._matchoutlook_cache:
            sim_h = self._match_similarity(home, mo.home_team)
            sim_a = self._match_similarity(away, mo.away_team)
            score = (sim_h + sim_a) / 2.0

            if score > 0.55 and score > best_score:
                best_score = score
                best_match = mo

        return best_match

    def find_scoregpt_match(self, home: str, away: str) -> Optional[ScoreGPTPrediction]:
        """Find ScoreGPT multi-model AI prediction matching home & away team"""
        best_match = None
        best_score = 0.0

        for sg in self._scoregpt_cache:
            sim_h = self._match_similarity(home, sg.home_team)
            sim_a = self._match_similarity(away, sg.away_team)
            score = (sim_h + sim_a) / 2.0

            if score > 0.55 and score > best_score:
                best_score = score
                best_match = sg

        return best_match

    def find_betensured_match(self, home: str, away: str) -> Optional[BetensuredPrediction]:
        """Find Betensured prediction matching home & away team"""
        best_match = None
        best_score = 0.0

        for be in self._betensured_cache:
            sim_h = self._match_similarity(home, be.home_team)
            sim_a = self._match_similarity(away, be.away_team)
            score = (sim_h + sim_a) / 2.0

            if score > 0.55 and score > best_score:
                best_score = score
                best_match = be

        return best_match

    def _calculate_form_points(self, form_list: List[str]) -> float:
        """Convert W/D/L form array to a 0.0 - 1.0 performance index"""
        if not form_list:
            return 0.5
        pts = 0
        for res in form_list:
            if res.upper() == "W":
                pts += 3
            elif res.upper() == "D":
                pts += 1
        max_pts = len(form_list) * 3
        return pts / max_pts if max_pts > 0 else 0.5

    def _parse_score(self, score_str: str) -> Tuple[float, float]:
        """Parse 'X-Y' scoreline into numeric (home, away) goals"""
        try:
            m = re.search(r'(\d+)\s*-\s*(\d+)', score_str)
            if m:
                return float(m.group(1)), float(m.group(2))
        except Exception:
            pass
        return 1.5, 1.2

    async def get_match_stats(self, home_team: str, away_team: str) -> AggregatedStats:
        """
        Aggregate multi-site statistics for a given match.
        """
        if not self._forebet_cache and not self._predictz_cache:
            await self.refresh_if_needed()

        match_str = f"{home_team} vs {away_team}"
        fb_pred = self.find_forebet_match(home_team, away_team)
        pz_pred = self.find_predictz_match(home_team, away_team)

        stats = AggregatedStats(
            match_str=match_str,
            home_team=home_team,
            away_team=away_team
        )

        # 1. Process Forebet
        if fb_pred:
            stats.has_forebet = True
            stats.forebet_p_home = fb_pred.prob_home
            stats.forebet_p_draw = fb_pred.prob_draw
            stats.forebet_p_away = fb_pred.prob_away
            stats.forebet_score = fb_pred.predicted_score
            stats.forebet_avg_goals = fb_pred.avg_goals

            h_goals, a_goals = self._parse_score(fb_pred.predicted_score)
            stats.est_home_xg = max(0.4, h_goals * 0.9 + 0.3)
            stats.est_away_xg = max(0.4, a_goals * 0.9 + 0.2)
            stats.est_home_xga = stats.est_away_xg
            stats.est_away_xga = stats.est_home_xg

        # 2. Process PredictZ
        if pz_pred:
            stats.has_predictz = True
            stats.predictz_outcome = pz_pred.predicted_outcome
            stats.predictz_score = pz_pred.predicted_score
            stats.home_form = pz_pred.home_form
            stats.away_form = pz_pred.away_form
            stats.home_form_factor = self._calculate_form_points(pz_pred.home_form)
            stats.away_form_factor = self._calculate_form_points(pz_pred.away_form)

        # 3. Process MatchOutlook
        mo_pred = self.find_matchoutlook_match(home_team, away_team)
        if mo_pred:
            stats.has_matchoutlook = True
            stats.matchoutlook_best_bet = mo_pred.best_bet
            stats.matchoutlook_desc = mo_pred.description

        # 4. Process ScoreGPT (Multi-Model AI Panel)
        sg_pred = self.find_scoregpt_match(home_team, away_team)
        if sg_pred:
            stats.has_scoregpt = True
            stats.scoregpt_pick = sg_pred.ai_pick
            stats.scoregpt_score = sg_pred.predicted_score
            stats.scoregpt_consensus = sg_pred.consensus_ratio
            stats.scoregpt_panel = sg_pred.panel_verdict

            if sg_pred.predicted_score and not stats.has_forebet:
                h_goals, a_goals = self._parse_score(sg_pred.predicted_score)
                stats.est_home_xg = max(0.5, h_goals * 0.9 + 0.3)
                stats.est_away_xg = max(0.5, a_goals * 0.9 + 0.2)
                stats.est_home_xga = stats.est_away_xg
                stats.est_away_xga = stats.est_home_xg

        # 5. Process Betensured
        be_pred = self.find_betensured_match(home_team, away_team)
        if be_pred:
            stats.has_betensured = True
            stats.betensured_tip = be_pred.tip
            stats.betensured_score = be_pred.predicted_score
            stats.betensured_p_home = be_pred.prob_home
            stats.betensured_p_draw = be_pred.prob_draw
            stats.betensured_p_away = be_pred.prob_away

            if be_pred.predicted_score and not stats.has_forebet and not (sg_pred and sg_pred.predicted_score):
                h_goals, a_goals = self._parse_score(be_pred.predicted_score)
                stats.est_home_xg = max(0.5, h_goals * 0.9 + 0.3)
                stats.est_away_xg = max(0.5, a_goals * 0.9 + 0.2)
                stats.est_home_xga = stats.est_away_xg
                stats.est_away_xga = stats.est_home_xg
            elif be_pred.prob_home > 0 and not stats.has_forebet:
                total_p = be_pred.prob_home + be_pred.prob_draw + be_pred.prob_away
                p_h = be_pred.prob_home / total_p
                p_a = be_pred.prob_away / total_p
                tot_xg = 2.5 + 0.6 * (p_h - p_a)
                stats.est_home_xg = round(max(0.5, tot_xg * (p_h + 0.5 * (be_pred.prob_draw / total_p))), 2)
                stats.est_away_xg = round(max(0.5, tot_xg * (p_a + 0.5 * (be_pred.prob_draw / total_p))), 2)
                stats.est_home_xga = stats.est_away_xg
                stats.est_away_xga = stats.est_home_xg

        # 3. Formulate Multi-Source Consensus Probabilities
        if stats.has_forebet and stats.has_predictz:
            # Weighted ensemble: 60% Forebet mathematical engine + 25% PredictZ scoreline + 15% Form factor
            pz_h = 0.55 if stats.predictz_outcome == "Home" else (0.25 if stats.predictz_outcome == "Draw" else 0.20)
            pz_d = 0.50 if stats.predictz_outcome == "Draw" else 0.25
            pz_a = 0.55 if stats.predictz_outcome == "Away" else (0.25 if stats.predictz_outcome == "Draw" else 0.20)

            c_home = (0.60 * stats.forebet_p_home) + (0.25 * pz_h) + (0.15 * stats.home_form_factor)
            c_draw = (0.60 * stats.forebet_p_draw) + (0.25 * pz_d) + (0.15 * 0.30)
            c_away = (0.60 * stats.forebet_p_away) + (0.25 * pz_a) + (0.15 * stats.away_form_factor)

            total = c_home + c_draw + c_away
            stats.consensus_home_prob = c_home / total
            stats.consensus_draw_prob = c_draw / total
            stats.consensus_away_prob = c_away / total

        elif stats.has_forebet:
            stats.consensus_home_prob = stats.forebet_p_home
            stats.consensus_draw_prob = stats.forebet_p_draw
            stats.consensus_away_prob = stats.forebet_p_away
        elif stats.has_predictz:
            pz_h = 0.50 if stats.predictz_outcome == "Home" else 0.25
            pz_d = 0.45 if stats.predictz_outcome == "Draw" else 0.30
            pz_a = 0.50 if stats.predictz_outcome == "Away" else 0.25
            total = pz_h + pz_d + pz_a
            stats.consensus_home_prob = pz_h / total
            stats.consensus_draw_prob = pz_d / total
            stats.consensus_away_prob = pz_a / total

        return stats

    def get_all_scraped_fixtures(self) -> List[Dict[str, Any]]:
        """
        Consolidate and deduplicate all scraped fixtures across all prediction platforms
        and SportyBet live bookmaker ground truth into a structured catalog for user selection.
        """
        if self._catalog_cache:
            return self._catalog_cache

        fixtures_map: Dict[str, Dict[str, Any]] = {}

        def get_key(home: str, away: str) -> str:
            h_norm = self._normalize_name(home)
            a_norm = self._normalize_name(away)
            return f"{h_norm}_vs_{a_norm}"

        def determine_recommended_outcome(oh: float, od: float, oa: float, tip_str: str = "") -> str:
            t = tip_str.lower().strip()
            if "away" in t or t == "2" or "0-1" in t or "0-2" in t or "1-2" in t or "0-3" in t:
                return "Away"
            if "home" in t or t == "1" or "1-0" in t or "2-0" in t or "2-1" in t or "3-0" in t or "3-1" in t:
                return "Home"
            if "draw" in t or t == "x" or "1-1" in t or "0-0" in t or "2-2" in t:
                return "Draw"
            
            # Market odds favorite
            if oa > 1.0 and oh > 1.0:
                if oa < oh and (od <= 1.0 or oa <= od):
                    return "Away"
                elif oh < oa and (od <= 1.0 or oh <= od):
                    return "Home"
                elif od > 1.0 and od < oh and od < oa:
                    return "Draw"
            return "Home"

        # Helper to match or create a fixture entry
        def find_existing_key(home: str, away: str) -> Optional[str]:
            for existing_key, item in fixtures_map.items():
                sim_h = self._match_similarity(home, item["home_team"])
                sim_a = self._match_similarity(away, item["away_team"])
                if (sim_h + sim_a) / 2.0 > 0.65:
                    return existing_key
            return None

        # 1. Ingest SportyBet fixtures FIRST as the official market ground truth
        for sb in self._sportybet_cache:
            if sb.odds_home <= 1.0 or sb.odds_away <= 1.0:
                continue
            key = get_key(sb.home_team, sb.away_team)
            oh = round(float(sb.odds_home), 2)
            od = round(float(sb.odds_draw), 2) if sb.odds_draw > 1.0 else 3.20
            oa = round(float(sb.odds_away), 2)
            rec = determine_recommended_outcome(oh, od, oa)
            fixtures_map[key] = {
                "id": key,
                "home_team": sb.home_team,
                "away_team": sb.away_team,
                "league": sb.league or "Football",
                "odds_home": oh,
                "odds_draw": od,
                "odds_away": oa,
                "sources": ["SportyBet"],
                "forebet_score": "",
                "forebet_probs": "",
                "scoregpt_pick": "",
                "betensured_tip": "",
                "predictz_tip": "",
                "matchoutlook_tip": "",
                "recommended_outcome": rec,
                "sporty_id": sb.sporty_id
            }

        # 2. Ingest Forebet fixtures
        for fb in self._forebet_cache:
            existing_key = find_existing_key(fb.home_team, fb.away_team)
            if existing_key:
                if "Forebet" not in fixtures_map[existing_key]["sources"]:
                    fixtures_map[existing_key]["sources"].append("Forebet")
                fixtures_map[existing_key]["forebet_score"] = fb.predicted_score
                fixtures_map[existing_key]["forebet_probs"] = f"H:{fb.prob_home}% D:{fb.prob_draw}% A:{fb.prob_away}%"
                if fb.tip in ["1", "X", "2"]:
                    fixtures_map[existing_key]["recommended_outcome"] = determine_recommended_outcome(
                        fixtures_map[existing_key]["odds_home"],
                        fixtures_map[existing_key]["odds_draw"],
                        fixtures_map[existing_key]["odds_away"],
                        fb.tip
                    )
            else:
                sb_match = self.find_sportybet_match(fb.home_team, fb.away_team)
                if sb_match and sb_match.odds_home > 1.0 and sb_match.odds_away > 1.0:
                    oh = round(float(sb_match.odds_home), 2)
                    od = round(float(sb_match.odds_draw), 2) if sb_match.odds_draw > 1.0 else 3.20
                    oa = round(float(sb_match.odds_away), 2)
                    srcs = ["SportyBet", "Forebet"]
                else:
                    oh = round(fb.odds_home, 2) if fb.odds_home > 1.0 else (round(100.0 / fb.prob_home, 2) if fb.prob_home > 0 else (1.90 if fb.prob_home > fb.prob_away else 2.60))
                    od = round(fb.odds_draw, 2) if fb.odds_draw > 1.0 else (round(100.0 / fb.prob_draw, 2) if fb.prob_draw > 0 else 3.30)
                    oa = round(fb.odds_away, 2) if fb.odds_away > 1.0 else (round(100.0 / fb.prob_away, 2) if fb.prob_away > 0 else (1.90 if fb.prob_away > fb.prob_home else 2.60))
                    srcs = ["Forebet"]

                key = get_key(fb.home_team, fb.away_team)
                rec = determine_recommended_outcome(oh, od, oa, fb.tip)
                fixtures_map[key] = {
                    "id": key,
                    "home_team": fb.home_team,
                    "away_team": fb.away_team,
                    "league": fb.league or "International",
                    "odds_home": oh,
                    "odds_draw": od,
                    "odds_away": oa,
                    "sources": srcs,
                    "forebet_score": fb.predicted_score,
                    "forebet_probs": f"H:{fb.prob_home}% D:{fb.prob_draw}% A:{fb.prob_away}%",
                    "scoregpt_pick": "",
                    "betensured_tip": "",
                    "predictz_tip": "",
                    "matchoutlook_tip": "",
                    "recommended_outcome": rec,
                    "sporty_id": sb_match.sporty_id if sb_match else ""
                }

        # 3. Ingest PredictZ
        for pz in self._predictz_cache:
            existing_key = find_existing_key(pz.home_team, pz.away_team)
            tip_text = f"{pz.predicted_outcome} ({pz.predicted_score})" if pz.predicted_score else pz.predicted_outcome
            if existing_key:
                if "PredictZ" not in fixtures_map[existing_key]["sources"]:
                    fixtures_map[existing_key]["sources"].append("PredictZ")
                fixtures_map[existing_key]["predictz_tip"] = tip_text
                if "SportyBet" not in fixtures_map[existing_key]["sources"] and pz.odds_home > 1.0 and pz.odds_away > 1.0:
                    fixtures_map[existing_key]["odds_home"] = round(pz.odds_home, 2)
                    fixtures_map[existing_key]["odds_draw"] = round(pz.odds_draw, 2)
                    fixtures_map[existing_key]["odds_away"] = round(pz.odds_away, 2)
                if pz.predicted_outcome in ["Home", "Draw", "Away"]:
                    fixtures_map[existing_key]["recommended_outcome"] = pz.predicted_outcome
            else:
                sb_match = self.find_sportybet_match(pz.home_team, pz.away_team)
                if sb_match and sb_match.odds_home > 1.0 and sb_match.odds_away > 1.0:
                    oh = round(float(sb_match.odds_home), 2)
                    od = round(float(sb_match.odds_draw), 2) if sb_match.odds_draw > 1.0 else 3.20
                    oa = round(float(sb_match.odds_away), 2)
                    srcs = ["SportyBet", "PredictZ"]
                elif pz.odds_home > 1.0 and pz.odds_away > 1.0:
                    oh = round(pz.odds_home, 2)
                    od = round(pz.odds_draw, 2) if pz.odds_draw > 1.0 else 3.30
                    oa = round(pz.odds_away, 2)
                    srcs = ["PredictZ"]
                else:
                    oh = 1.85 if pz.predicted_outcome == "Home" else (3.60 if pz.predicted_outcome == "Away" else 2.60)
                    od = 3.30 if pz.predicted_outcome != "Draw" else 2.90
                    oa = 1.95 if pz.predicted_outcome == "Away" else (3.80 if pz.predicted_outcome == "Home" else 2.60)
                    srcs = ["PredictZ"]

                key = get_key(pz.home_team, pz.away_team)
                rec = pz.predicted_outcome if pz.predicted_outcome in ["Home", "Draw", "Away"] else determine_recommended_outcome(oh, od, oa, tip_text)
                fixtures_map[key] = {
                    "id": key,
                    "home_team": pz.home_team,
                    "away_team": pz.away_team,
                    "league": pz.league or "League",
                    "odds_home": oh,
                    "odds_draw": od,
                    "odds_away": oa,
                    "sources": srcs,
                    "forebet_score": "",
                    "forebet_probs": "",
                    "scoregpt_pick": "",
                    "betensured_tip": "",
                    "predictz_tip": tip_text,
                    "matchoutlook_tip": "",
                    "recommended_outcome": rec,
                    "sporty_id": sb_match.sporty_id if sb_match else ""
                }

        # 4. Ingest ScoreGPT
        for sg in self._scoregpt_cache:
            existing_key = find_existing_key(sg.home_team, sg.away_team)
            tip_text = f"{sg.ai_pick} ({sg.consensus_ratio or sg.panel_verdict})" if sg.ai_pick else (sg.consensus_ratio or sg.panel_verdict)
            if existing_key:
                if "ScoreGPT" not in fixtures_map[existing_key]["sources"]:
                    fixtures_map[existing_key]["sources"].append("ScoreGPT")
                fixtures_map[existing_key]["scoregpt_pick"] = tip_text
                if sg.ai_pick in ["1", "X", "2", "Home", "Away", "Draw"] and not fixtures_map[existing_key].get("recommended_outcome"):
                    fixtures_map[existing_key]["recommended_outcome"] = determine_recommended_outcome(
                        fixtures_map[existing_key]["odds_home"],
                        fixtures_map[existing_key]["odds_draw"],
                        fixtures_map[existing_key]["odds_away"],
                        sg.ai_pick
                    )
            else:
                sb_match = self.find_sportybet_match(sg.home_team, sg.away_team)
                if sb_match and sb_match.odds_home > 1.0 and sb_match.odds_away > 1.0:
                    oh = round(float(sb_match.odds_home), 2)
                    od = round(float(sb_match.odds_draw), 2) if sb_match.odds_draw > 1.0 else 3.20
                    oa = round(float(sb_match.odds_away), 2)
                    srcs = ["SportyBet", "ScoreGPT"]
                else:
                    oh, od, oa = 2.10, 3.20, 3.10
                    srcs = ["ScoreGPT"]
                key = get_key(sg.home_team, sg.away_team)
                rec = determine_recommended_outcome(oh, od, oa, sg.ai_pick)
                fixtures_map[key] = {
                    "id": key,
                    "home_team": sg.home_team,
                    "away_team": sg.away_team,
                    "league": sg.league or "Global",
                    "odds_home": oh,
                    "odds_draw": od,
                    "odds_away": oa,
                    "sources": srcs,
                    "forebet_score": "",
                    "forebet_probs": "",
                    "scoregpt_pick": tip_text,
                    "betensured_tip": "",
                    "predictz_tip": "",
                    "matchoutlook_tip": "",
                    "recommended_outcome": rec,
                    "sporty_id": sb_match.sporty_id if sb_match else ""
                }

        # 5. Ingest Betensured
        for be in self._betensured_cache:
            existing_key = find_existing_key(be.home_team, be.away_team)
            tip_text = f"{be.tip} ({be.predicted_score})" if be.predicted_score else be.tip
            if existing_key:
                if "Betensured" not in fixtures_map[existing_key]["sources"]:
                    fixtures_map[existing_key]["sources"].append("Betensured")
                fixtures_map[existing_key]["betensured_tip"] = tip_text
                if "SportyBet" not in fixtures_map[existing_key]["sources"] and be.odds_home > 1.0:
                    fixtures_map[existing_key]["odds_home"] = round(be.odds_home, 2)
                    fixtures_map[existing_key]["odds_draw"] = round(be.odds_draw, 2) if be.odds_draw > 1.0 else 3.20
                    fixtures_map[existing_key]["odds_away"] = round(be.odds_away, 2)
            else:
                sb_match = self.find_sportybet_match(be.home_team, be.away_team)
                if sb_match and sb_match.odds_home > 1.0 and sb_match.odds_away > 1.0:
                    oh = round(float(sb_match.odds_home), 2)
                    od = round(float(sb_match.odds_draw), 2) if sb_match.odds_draw > 1.0 else 3.20
                    oa = round(float(sb_match.odds_away), 2)
                    srcs = ["SportyBet", "Betensured"]
                else:
                    oh = round(be.odds_home, 2) if be.odds_home > 1.0 else (round(100.0 / be.prob_home, 2) if be.prob_home > 0 else 2.10)
                    od = round(be.odds_draw, 2) if be.odds_draw > 1.0 else (round(100.0 / be.prob_draw, 2) if be.prob_draw > 0 else 3.20)
                    oa = round(be.odds_away, 2) if be.odds_away > 1.0 else (round(100.0 / be.prob_away, 2) if be.prob_away > 0 else 3.10)
                    srcs = ["Betensured"]

                key = get_key(be.home_team, be.away_team)
                rec = determine_recommended_outcome(oh, od, oa, be.tip)
                fixtures_map[key] = {
                    "id": key,
                    "home_team": be.home_team,
                    "away_team": be.away_team,
                    "league": be.league or "Tournament",
                    "odds_home": oh,
                    "odds_draw": od,
                    "odds_away": oa,
                    "sources": srcs,
                    "forebet_score": "",
                    "forebet_probs": "",
                    "scoregpt_pick": "",
                    "betensured_tip": tip_text,
                    "predictz_tip": "",
                    "matchoutlook_tip": "",
                    "recommended_outcome": rec,
                    "sporty_id": sb_match.sporty_id if sb_match else ""
                }

        # 6. Ingest MatchOutlook
        for mo in self._matchoutlook_cache:
            matched_key = find_existing_key(mo.home_team, mo.away_team)
            if matched_key:
                if "MatchOutlook" not in fixtures_map[matched_key]["sources"]:
                    fixtures_map[matched_key]["sources"].append("MatchOutlook")
                fixtures_map[matched_key]["matchoutlook_tip"] = mo.best_bet

        fixture_list = list(fixtures_map.values())
        # Sort so fixtures with SportyBet & high source consensus appear at the top
        fixture_list.sort(key=lambda x: (1 if "SportyBet" in x["sources"] else 0, len(x["sources"]), x["league"]), reverse=True)
        self._catalog_cache = fixture_list
        return fixture_list


# Global default instance
_stats_aggregator = None

def get_stats_aggregator() -> StatsAggregator:
    global _stats_aggregator
    if _stats_aggregator is None:
        _stats_aggregator = StatsAggregator()
    return _stats_aggregator
