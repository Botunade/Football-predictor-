"""
Feature Builder Module
Transforms raw match data into structured features for ML models
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TeamFeatures:
    """Features for a single team"""
    xg: float = 1.5
    xga: float = 1.2
    xgd: float = 0.3
    ppda: float = 10.5
    possession: float = 50.0
    injury_index: float = 0.1
    fatigue_index: float = 0.2

    def to_dict(self) -> Dict[str, float]:
        return {
            "xG": self.xg,
            "xGA": self.xga,
            "xGD": self.xgd,
            "PPDA": self.ppda,
            "possession": self.possession,
            "injury_index": self.injury_index,
            "fatigue_index": self.fatigue_index
        }


@dataclass
class MatchFeatures:
    """Complete match features for prediction"""
    match_id: str
    home_team: str
    away_team: str
    home_features: TeamFeatures
    away_features: TeamFeatures

    # H2H features
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_xga: float = 0.0
    away_xga: float = 0.0
    home_xgd: float = 0.0
    away_xgd: float = 0.0
    home_ppda: float = 0.0
    away_ppda: float = 0.0
    home_injury: float = 0.0
    away_injury: float = 0.0
    home_fatigue: float = 0.0
    away_fatigue: float = 0.0
    home_away_advantage: float = 1.1
    rest_days_diff: int = 0
    is_derby: int = 0

    # Odds
    odds_home: float = 0.0
    odds_draw: float = 0.0
    odds_away: float = 0.0

    # Market data
    line_movement: float = 0.05
    public_sentiment: float = 0.6
    betting_volume: int = 1000000

    # Scraped Consensus Intelligence (Forebet + PredictZ)
    consensus_home_prob: float = 0.33
    consensus_draw_prob: float = 0.33
    consensus_away_prob: float = 0.34
    forebet_score: str = ""
    predictz_score: str = ""

    # Phase 2 Enrichment: Direct H2H, League Table Motivation, and Sharp Money
    h2h_data: Dict[str, Any] = field(default_factory=dict)
    league_context: Dict[str, Any] = field(default_factory=dict)
    sharp_money: Dict[str, Any] = field(default_factory=dict)

    def to_list(self, feature_columns: List[str] = None) -> List[float]:
        """Convert to ordered list for model input"""
        if feature_columns is None:
            feature_columns = [
                "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
                "home_ppda", "away_ppda", "home_injury", "away_injury",
                "home_fatigue", "away_fatigue", "home_away_advantage",
                "rest_days_diff", "is_derby"
            ]

        mapping = {
            "home_xG": self.home_xg,
            "away_xG": self.away_xg,
            "home_xGA": self.home_xga,
            "away_xGA": self.away_xga,
            "home_xGD": self.home_xgd,
            "away_xGD": self.away_xgd,
            "home_ppda": self.home_ppda,
            "away_ppda": self.away_ppda,
            "home_injury": self.home_injury,
            "away_injury": self.away_injury,
            "home_fatigue": self.home_fatigue,
            "away_fatigue": self.away_fatigue,
            "home_away_advantage": self.home_away_advantage,
            "rest_days_diff": self.rest_days_diff,
            "is_derby": self.is_derby,
            "odds_home": self.odds_home,
            "odds_draw": self.odds_draw,
            "odds_away": self.odds_away,
        }

        return [mapping.get(col, 0.0) for col in feature_columns]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "match_id": self.match_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "home_xg": self.home_xg,
            "away_xg": self.away_xg,
            "home_xG": self.home_xg,
            "away_xG": self.away_xg,
            "home_xga": self.home_xga,
            "away_xga": self.away_xga,
            "home_xGA": self.home_xga,
            "away_xGA": self.away_xga,
            "home_xgd": self.home_xgd,
            "away_xgd": self.away_xgd,
            "home_ppda": self.home_ppda,
            "away_ppda": self.away_ppda,
            "home_injury": self.home_injury,
            "away_injury": self.away_injury,
            "home_fatigue": self.home_fatigue,
            "away_fatigue": self.away_fatigue,
            "home_away_advantage": self.home_away_advantage,
            "rest_days_diff": self.rest_days_diff,
            "is_derby": self.is_derby,
            "odds_home": self.odds_home,
            "odds_draw": self.odds_draw,
            "odds_away": self.odds_away,
            "line_movement": self.line_movement,
            "public_sentiment": self.public_sentiment,
            "betting_volume": self.betting_volume,
            "consensus_home_prob": round(self.consensus_home_prob, 4),
            "consensus_draw_prob": round(self.consensus_draw_prob, 4),
            "consensus_away_prob": round(self.consensus_away_prob, 4),
            "forebet_score": self.forebet_score,
            "predictz_score": self.predictz_score,
            "h2h_data": self.h2h_data,
            "league_context": self.league_context,
            "sharp_money": self.sharp_money
        }


class FeatureBuilder:
    """Advanced feature builder with caching and async support"""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self._cache: Dict[str, MatchFeatures] = {}

    def reset_cache(self):
        """Clear the feature cache"""
        self._cache.clear()

    def is_derby(self, home_team: str, away_team: str) -> int:
        """Check if match is a derby"""
        derby_pairs = [
            ("arsenal", "chelsea"),
            ("liverpool", "everton"),
            ("real Madrid", "barcelona"),
            ("bayern Munich", "borussia Dortmund"),
            # Add more derby pairs as needed
        ]

        pair = (home_team.lower(), away_team.lower())
        reverse_pair = (away_team.lower(), home_team.lower())
        return 1 if pair in derby_pairs or reverse_pair in derby_pairs else 0

    def compute_context(self, fixture: Dict) -> Dict:
        """Compute contextual metrics"""
        from difflib import get_close_matches

        return {
            "home_away_advantage": 1.1,
            "rest_days_diff": 0,
            "is_derby": self.is_derby(
                fixture.get("home_team", ""),
                fixture.get("away_team", "")
            ),
            "weather_impact": 0,
            "pitch_quality": 1,
            "travel_distance": 50,
            "match_importance": 0.8
        }

    def fetch_market_data(self, home_team: str, away_team: str) -> Dict:
        """Fetch market signals - can be overridden with real data"""
        return {
            "line_movement": 0.05,
            "public_sentiment": 0.6,
            "betting_volume": 1000000
        }

    async def extract_team_features(self, team_name: str, sport: str = "football") -> TeamFeatures:
        """Extract advanced team features"""
        # Default values - can be enhanced with real data scraping
        defaults = {
            "football": TeamFeatures(xg=1.5, xga=1.2, xgd=0.3, ppda=10.5, possession=50.0),
            "basketball": TeamFeatures(xg=110.0, xga=108.0, xgd=2.0, ppda=22.0, possession=48.0),
            "hockey": TeamFeatures(xg=3.2, xga=2.8, xgd=0.4, ppda=15.0, possession=52.0)
        }

        return defaults.get(sport, defaults["football"])

    async def build_features(self, fixture: Dict, sport: str = "football") -> MatchFeatures:
        """
        Build complete feature set for a match

        Args:
            fixture: Match data dictionary
            sport: Sport type (football, basketball, hockey)

        Returns:
            MatchFeatures object with all features
        """
        # Generate unique match ID
        home = fixture.get("home_team", "Home")
        away = fixture.get("away_team", "Away")
        match_id = f"{home}_vs_{away}_{datetime.now(timezone.utc).isoformat()}"

        # Extract team IDs and names
        teams = fixture.get("teams")
        has_teams_data = (teams and
                         isinstance(teams, dict) and
                         teams.get("home") is not None and
                         teams.get("away") is not None)

        if has_teams_data:
            h_id = fixture["teams"].get("home", {}).get("id", 0)
            a_id = fixture["teams"].get("away", {}).get("id", 0)
            h_name = fixture["teams"].get("home", {}).get("name", home)
            a_name = fixture["teams"].get("away", {}).get("name", away)
            l_id = fixture.get("league", {}).get("id", 39)
            season = fixture.get("league", {}).get("season", 2024)
        else:
            h_name = home
            a_name = away
            h_id = fixture.get("home_id", 0)
            a_id = fixture.get("away_id", 0)
            l_id = fixture.get("league_id", 39)
            season = fixture.get("season", 2024)

        # Extract team features in parallel
        home_features = await self.extract_team_features(h_name, sport)
        away_features = await self.extract_team_features(a_name, sport)

        # Extract odds with defaults
        odds_home = float(fixture.get("odds_home", 0.0))
        odds_draw = float(fixture.get("odds_draw", 0.0))
        odds_away = float(fixture.get("odds_away", 0.0))

        # Query live scraped stats (Forebet, PredictZ, MatchOutlook, ScoreGPT, Betensured) for football
        c_h, c_d, c_a = 0.33, 0.33, 0.34
        fb_score, pz_score = "", ""
        if sport == "football":
            try:
                try:
                    from ..data_extractor.stats_scrapers import get_stats_aggregator
                except (ImportError, ValueError):
                    try:
                        from data_extractor.stats_scrapers import get_stats_aggregator
                    except ImportError:
                        from src.data_extractor.stats_scrapers import get_stats_aggregator
                agg = get_stats_aggregator()
                stats = await agg.get_match_stats(h_name, a_name)
                if stats and (stats.has_forebet or stats.has_predictz or stats.has_scoregpt or stats.has_betensured):
                    home_features.xg = stats.est_home_xg
                    home_features.xga = stats.est_home_xga
                    home_features.xgd = round(stats.est_home_xg - stats.est_home_xga, 2)
                    away_features.xg = stats.est_away_xg
                    away_features.xga = stats.est_away_xga
                    away_features.xgd = round(stats.est_away_xg - stats.est_away_xga, 2)
                    c_h = stats.consensus_home_prob
                    c_d = stats.consensus_draw_prob
                    c_a = stats.consensus_away_prob
                    fb_score = stats.forebet_score or stats.scoregpt_score or stats.betensured_score
                    pz_score = stats.predictz_score
                elif odds_home > 1.0 and odds_away > 1.0:
                    r_h = 1.0 / odds_home
                    r_d = 1.0 / (odds_draw if odds_draw > 1.0 else 3.20)
                    r_a = 1.0 / odds_away
                    s = r_h + r_d + r_a
                    p_h, p_d, p_a = r_h / s, r_d / s, r_a / s
                    tot_xg = max(1.8, min(3.8, 2.5 + 0.6 * (p_h - p_a)))
                    home_features.xg = round(tot_xg * (p_h + 0.5 * p_d), 2)
                    away_features.xg = round(tot_xg * (p_a + 0.5 * p_d), 2)
                    home_features.xga = away_features.xg
                    away_features.xga = home_features.xg
                    home_features.xgd = round(home_features.xg - home_features.xga, 2)
                    away_features.xgd = round(away_features.xg - away_features.xga, 2)
                    c_h, c_d, c_a = p_h, p_d, p_a
            except Exception as e:
                logger.debug(f"Stats aggregator integration skipped: {e}")

        # Build context and market data
        context = self.compute_context(fixture)
        market = self.fetch_market_data(h_name, a_name)
        if c_h != 0.33:
            market["public_sentiment"] = c_h

        # Phase 2 Enrichment: Direct H2H, Table Standings Motivation, Sharp Money Signals
        h2h_dict, league_mot_dict, sharp_dict = {}, {}, {}
        if sport == "football":
            try:
                try:
                    from ..data_extractor.enrichment_engine import DataEnrichmentEngine
                except (ImportError, ValueError):
                    try:
                        from data_extractor.enrichment_engine import DataEnrichmentEngine
                    except ImportError:
                        from src.data_extractor.enrichment_engine import DataEnrichmentEngine

                h2h_summary = DataEnrichmentEngine.analyze_h2h(
                    home_team=h_name,
                    away_team=a_name,
                    odds_home=odds_home,
                    odds_away=odds_away
                )
                h2h_dict = h2h_summary.to_dict()

                league_mot = DataEnrichmentEngine.evaluate_league_context(
                    home_team=h_name,
                    away_team=a_name,
                    league=str(fixture.get("league", "General")),
                    odds_home=odds_home,
                    odds_away=odds_away
                )
                league_mot_dict = league_mot.to_dict()

                fair_price = 1.0 / c_h if c_h > 0 else (odds_home if odds_home > 0 else 2.0)
                sharp_sig = DataEnrichmentEngine.track_sharp_money(
                    home_team=h_name,
                    away_team=a_name,
                    bookmaker_odds=odds_home if odds_home > 0 else 2.0,
                    fair_odds=fair_price,
                    model_prob=c_h,
                    selected_outcome=str(fixture.get("selected_outcome", "Home"))
                )
                sharp_dict = sharp_sig.to_dict()
            except Exception as e:
                logger.debug(f"Enrichment engine error: {e}")

        # Create feature object
        features = MatchFeatures(
            match_id=match_id,
            home_team=h_name,
            away_team=a_name,
            home_features=home_features,
            away_features=away_features,
            home_xg=home_features.xg,
            away_xg=away_features.xg,
            home_xga=home_features.xga,
            away_xga=away_features.xga,
            home_xgd=home_features.xgd,
            away_xgd=away_features.xgd,
            home_ppda=home_features.ppda,
            away_ppda=away_features.ppda,
            home_injury=home_features.injury_index,
            away_injury=away_features.injury_index,
            home_fatigue=home_features.fatigue_index,
            away_fatigue=away_features.fatigue_index,
            home_away_advantage=context["home_away_advantage"],
            rest_days_diff=context["rest_days_diff"],
            is_derby=context["is_derby"],
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            line_movement=market["line_movement"],
            public_sentiment=market["public_sentiment"],
            betting_volume=market["betting_volume"],
            consensus_home_prob=c_h,
            consensus_draw_prob=c_d,
            consensus_away_prob=c_a,
            forebet_score=fb_score,
            predictz_score=pz_score,
            h2h_data=h2h_dict,
            league_context=league_mot_dict,
            sharp_money=sharp_dict
        )

        # Cache if enabled
        if self.use_cache:
            self._cache[match_id] = features

        return features


async def build_features(fixture: Dict, sport: str = "football") -> MatchFeatures:
    """
    Convenience function for building features
    """
    builder = FeatureBuilder()
    return await builder.build_features(fixture, sport)


async def extract_team_features(team_name: str, sport: str = "football") -> TeamFeatures:
    """
    Convenience function for extracting team features
    """
    builder = FeatureBuilder()
    return await builder.extract_team_features(team_name, sport)