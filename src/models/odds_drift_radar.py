"""
Odds Drift & Sharp Money Steam Radar
Tracks odds movements over time to detect sharp money syndicate action,
steam moves (rapidly shortening odds), and market drift (fading lines).
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("OddsDriftRadar")
logger.setLevel(logging.INFO)

RADAR_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "odds_radar_snapshots.json"


@dataclass
class DriftSignal:
    home_team: str
    away_team: str
    open_odds_h: float
    current_odds_h: float
    drift_h_pct: float
    open_odds_a: float
    current_odds_a: float
    drift_a_pct: float
    steam_target: Optional[str] = None
    signal_type: str = "NEUTRAL"
    confidence: float = 0.5
    narrative: str = "Market pricing remains stable."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "home_team": self.home_team,
            "away_team": self.away_team,
            "open_odds_h": round(self.open_odds_h, 2),
            "current_odds_h": round(self.current_odds_h, 2),
            "drift_h_pct": round(self.drift_h_pct, 1),
            "open_odds_a": round(self.open_odds_a, 2),
            "current_odds_a": round(self.current_odds_a, 2),
            "drift_a_pct": round(self.drift_a_pct, 1),
            "steam_target": self.steam_target,
            "signal_type": self.signal_type,
            "confidence": round(self.confidence, 2),
            "narrative": self.narrative
        }


class OddsDriftRadar:
    """
    Maintains persistent historical price logs and identifies sharp market moves.
    """

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or RADAR_STORE_PATH
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List[Dict[str, Any]]] = self._load_store()

    def _normalize_key(self, home: str, away: str) -> str:
        return f"{home.lower().strip()}_vs_{away.lower().strip()}"

    def _load_store(self) -> Dict[str, List[Dict[str, Any]]]:
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except Exception as e:
                logger.debug(f"Failed to load drift store: {e}")
        return {}

    def _save_store(self):
        try:
            self.store_path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to save drift store: {e}")

    def record_snapshot(self, home: str, away: str, oh: float, od: float, oa: float) -> DriftSignal:
        """
        Records a new price observation and computes instant drift signal.
        """
        key = self._normalize_key(home, away)
        now_ts = int(time.time())

        if key not in self.history:
            self.history[key] = []

        entry = {
            "ts": now_ts,
            "oh": round(float(oh), 2),
            "od": round(float(od), 2),
            "oa": round(float(oa), 2)
        }
        self.history[key].append(entry)
        if len(self.history[key]) > 20:
            self.history[key] = self.history[key][-20:]

        self._save_store()
        return self.analyze_drift(home, away, oh, od, oa)

    def analyze_drift(self, home: str, away: str, current_oh: float, current_od: float, current_oa: float) -> DriftSignal:
        """
        Analyzes historical delta between opening recorded odds and current market prices.
        """
        key = self._normalize_key(home, away)
        snapshots = self.history.get(key, [])

        if not snapshots or len(snapshots) < 2:
            return DriftSignal(
                home_team=home,
                away_team=away,
                open_odds_h=current_oh,
                current_odds_h=current_oh,
                drift_h_pct=0.0,
                open_odds_a=current_oa,
                current_odds_a=current_oa,
                drift_a_pct=0.0,
                signal_type="INITIAL_BASELINE",
                narrative="Opening price baseline recorded."
            )

        open_entry = snapshots[0]
        open_oh = open_entry.get("oh", current_oh)
        open_oa = open_entry.get("oa", current_oa)

        # Delta percentages (negative means odds dropped / shortened = sharp money incoming)
        dh_pct = ((current_oh - open_oh) / open_oh) * 100.0 if open_oh > 0 else 0.0
        da_pct = ((current_oa - open_oa) / open_oa) * 100.0 if open_oa > 0 else 0.0

        # STEAM MOVE threshold: odds shortened by > 5.5%
        if dh_pct <= -5.5:
            return DriftSignal(
                home_team=home,
                away_team=away,
                open_odds_h=open_oh,
                current_odds_h=current_oh,
                drift_h_pct=dh_pct,
                open_odds_a=open_oa,
                current_odds_a=current_oa,
                drift_a_pct=da_pct,
                steam_target="Home",
                signal_type="STEAM_BUY_HOME",
                confidence=min(0.92, 0.65 + abs(dh_pct) / 50.0),
                narrative=f"⚡ STEAM MOVE: Sharp syndicate money poured into {home}. Odds crashed from {open_oh:.2f} to {current_oh:.2f} ({dh_pct:+.1f}%)."
            )
        elif da_pct <= -5.5:
            return DriftSignal(
                home_team=home,
                away_team=away,
                open_odds_h=open_oh,
                current_odds_h=current_oh,
                drift_h_pct=dh_pct,
                open_odds_a=open_oa,
                current_odds_a=current_oa,
                drift_a_pct=da_pct,
                steam_target="Away",
                signal_type="STEAM_BUY_AWAY",
                confidence=min(0.92, 0.65 + abs(da_pct) / 50.0),
                narrative=f"⚡ STEAM MOVE: Heavy sharp flow detected on {away}. Odds shortened from {open_oa:.2f} to {current_oa:.2f} ({da_pct:+.1f}%)."
            )
        elif dh_pct >= 8.0:
            return DriftSignal(
                home_team=home,
                away_team=away,
                open_odds_h=open_oh,
                current_odds_h=current_oh,
                drift_h_pct=dh_pct,
                open_odds_a=open_oa,
                current_odds_a=current_oa,
                drift_a_pct=da_pct,
                signal_type="FADE_HOME",
                confidence=0.70,
                narrative=f"📉 DRIFT: Smart money fading {home}. Odds lengthened from {open_oh:.2f} to {current_oh:.2f} ({dh_pct:+.1f}%)."
            )
        elif da_pct >= 8.0:
            return DriftSignal(
                home_team=home,
                away_team=away,
                open_odds_h=open_oh,
                current_odds_h=current_oh,
                drift_h_pct=dh_pct,
                open_odds_a=open_oa,
                current_odds_a=current_oa,
                drift_a_pct=da_pct,
                signal_type="FADE_AWAY",
                confidence=0.70,
                narrative=f"📉 DRIFT: Smart money fading {away}. Odds lengthened from {open_oa:.2f} to {current_oa:.2f} ({da_pct:+.1f}%)."
            )

        return DriftSignal(
            home_team=home,
            away_team=away,
            open_odds_h=open_oh,
            current_odds_h=current_oh,
            drift_h_pct=dh_pct,
            open_odds_a=open_oa,
            current_odds_a=current_oa,
            drift_a_pct=da_pct,
            signal_type="STABLE",
            confidence=0.55,
            narrative="Market prices are balanced with normal recreational trading volume."
        )


_radar_instance = None


def get_odds_drift_radar() -> OddsDriftRadar:
    global _radar_instance
    if _radar_instance is None:
        _radar_instance = OddsDriftRadar()
    return _radar_instance
