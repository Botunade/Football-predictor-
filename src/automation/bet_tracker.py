"""
Automated Bet Tracker & P&L Journal
Tracks placed wagers, dry-run validations, settlement outcomes, net profit/loss,
ROI metrics, and market-by-market strike rate analytics.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("BetTracker")
logger.setLevel(logging.INFO)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bet_tracker.db"


class BetTracker:
    """
    Manages the persistent betting ledger, calculates quantitative performance metrics,
    and tracks winning strike rates across individual betting markets.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bets (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticket_id TEXT,
                        booking_code TEXT,
                        timestamp TEXT,
                        stake REAL,
                        combined_odds REAL,
                        potential_payout REAL,
                        status TEXT DEFAULT 'PENDING',
                        payout REAL DEFAULT 0.0,
                        profit REAL DEFAULT 0.0,
                        is_dry_run INTEGER DEFAULT 0,
                        notes TEXT,
                        created_at TEXT,
                        settled_at TEXT
                    )
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bet_legs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        bet_id INTEGER,
                        home_team TEXT,
                        away_team TEXT,
                        league TEXT,
                        sporty_id TEXT,
                        market_name TEXT,
                        selection TEXT,
                        odds REAL,
                        status TEXT DEFAULT 'PENDING',
                        final_score TEXT,
                        FOREIGN KEY(bet_id) REFERENCES bets(id) ON DELETE CASCADE
                    )
                """)
        finally:
            conn.close()

    def record_bet(
        self,
        booking_code: str,
        stake: float,
        combined_odds: float,
        legs: List[Dict[str, Any]],
        ticket_id: Optional[str] = None,
        is_dry_run: bool = False,
        notes: str = ""
    ) -> int:
        """
        Record a newly placed or dry-run accumulator ticket with all its individual legs.
        """
        now_str = datetime.now().isoformat()
        potential_payout = round(stake * combined_odds, 2)
        initial_status = "DRY_RUN" if is_dry_run else "PENDING"

        conn = self._get_connection()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO bets (
                        ticket_id, booking_code, timestamp, stake, combined_odds,
                        potential_payout, status, payout, profit, is_dry_run, notes,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 0.0, ?, ?, ?)
                """, (
                    ticket_id or "",
                    booking_code.upper().strip() if booking_code else "",
                    now_str,
                    stake,
                    combined_odds,
                    potential_payout,
                    initial_status,
                    1 if is_dry_run else 0,
                    notes,
                    now_str
                ))
                bet_id = cur.lastrowid

                for leg in legs:
                    cur.execute("""
                        INSERT INTO bet_legs (
                            bet_id, home_team, away_team, league, sporty_id,
                            market_name, selection, odds, status, final_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', '')
                    """, (
                        bet_id,
                        leg.get("home_team", ""),
                        leg.get("away_team", ""),
                        leg.get("league", ""),
                        leg.get("sporty_id", ""),
                        leg.get("market_name", "Match Outcome"),
                        leg.get("selection") or leg.get("selected_outcome", ""),
                        float(leg.get("odds") or leg.get("selected_odds") or 1.0)
                    ))
                logger.info(f"Recorded bet #{bet_id} [{booking_code}] ({len(legs)} legs, stake: {stake} NGN, status: {initial_status})")
                return bet_id
        finally:
            conn.close()

    def settle_bet(
        self,
        bet_id: int,
        status: str,
        actual_payout: Optional[float] = None,
        leg_results: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Settles a bet as WON, LOST, or VOID, updating net profit.
        """
        status = status.upper().strip()
        now_str = datetime.now().isoformat()

        conn = self._get_connection()
        try:
            with conn:
                cur = conn.cursor()
                cur.execute("SELECT stake, potential_payout FROM bets WHERE id = ?", (bet_id,))
                row = cur.fetchone()
                if not row:
                    return False

                stake = row["stake"]
                potential = row["potential_payout"]

                if actual_payout is None:
                    if status == "WON":
                        payout = potential
                    elif status == "VOID":
                        payout = stake
                    else:
                        payout = 0.0
                else:
                    payout = actual_payout

                profit = round(payout - stake, 2)

                cur.execute("""
                    UPDATE bets
                    SET status = ?, payout = ?, profit = ?, settled_at = ?
                    WHERE id = ?
                """, (status, payout, profit, now_str, bet_id))

                if leg_results:
                    for leg in leg_results:
                        leg_id = leg.get("id")
                        l_status = leg.get("status", "PENDING").upper()
                        l_score = leg.get("final_score", "")
                        if leg_id:
                            cur.execute("""
                                UPDATE bet_legs
                                SET status = ?, final_score = ?
                                WHERE id = ? AND bet_id = ?
                            """, (l_status, l_score, leg_id, bet_id))
                return True
        finally:
            conn.close()

    def get_bets(self, limit: int = 50, include_dry_runs: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieves all recorded tickets with nested leg information.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            if include_dry_runs:
                cur.execute("SELECT * FROM bets ORDER BY id DESC LIMIT ?", (limit,))
            else:
                cur.execute("SELECT * FROM bets WHERE is_dry_run = 0 ORDER BY id DESC LIMIT ?", (limit,))
            rows = cur.fetchall()

            bets = []
            for r in rows:
                bet_dict = dict(r)
                cur.execute("SELECT * FROM bet_legs WHERE bet_id = ?", (bet_dict["id"],))
                bet_dict["legs"] = [dict(leg) for leg in cur.fetchall()]
                bets.append(bet_dict)
            return bets
        finally:
            conn.close()

    def get_pnl_summary(self, include_dry_runs: bool = False) -> Dict[str, Any]:
        """
        Computes financial summary: ROI, Win Rate, Net Profit, and Market-by-Market breakdown.
        """
        conn = self._get_connection()
        try:
            cur = conn.cursor()
            where_clause = "" if include_dry_runs else "WHERE is_dry_run = 0"
            cur.execute(f"SELECT * FROM bets {where_clause}")
            bets = [dict(r) for r in cur.fetchall()]

            total_staked = 0.0
            total_payout = 0.0
            won_count = 0
            lost_count = 0
            pending_count = 0
            void_count = 0
            dry_run_count = 0

            for b in bets:
                st = b.get("status")
                if b.get("is_dry_run"):
                    dry_run_count += 1
                if st == "WON":
                    won_count += 1
                    total_staked += b.get("stake", 0.0)
                    total_payout += b.get("payout", 0.0)
                elif st == "LOST":
                    lost_count += 1
                    total_staked += b.get("stake", 0.0)
                elif st == "VOID":
                    void_count += 1
                    total_staked += b.get("stake", 0.0)
                    total_payout += b.get("payout", 0.0)
                elif st == "PENDING":
                    pending_count += 1
                    total_staked += b.get("stake", 0.0)

            settled_count = won_count + lost_count
            net_profit = round(total_payout - total_staked, 2)
            roi_pct = round((net_profit / total_staked * 100), 2) if total_staked > 0 else 0.0
            win_rate_pct = round((won_count / settled_count * 100), 2) if settled_count > 0 else 0.0

            # Market-by-Market Breakdown
            cur.execute("""
                SELECT market_name, selection, status, COUNT(*) as cnt
                FROM bet_legs
                GROUP BY market_name, selection, status
            """)
            market_stats: Dict[str, Dict[str, Any]] = {}
            for row in cur.fetchall():
                m_name = row["selection"] or row["market_name"] or "Unknown"
                st = row["status"]
                cnt = row["cnt"]
                if m_name not in market_stats:
                    market_stats[m_name] = {"won": 0, "lost": 0, "pending": 0, "total": 0, "strike_rate_pct": 0.0}
                if st == "WON":
                    market_stats[m_name]["won"] += cnt
                elif st == "LOST":
                    market_stats[m_name]["lost"] += cnt
                else:
                    market_stats[m_name]["pending"] += cnt
                market_stats[m_name]["total"] += cnt

            for m_name, d in market_stats.items():
                settled_legs = d["won"] + d["lost"]
                d["strike_rate_pct"] = round((d["won"] / settled_legs * 100), 1) if settled_legs > 0 else 0.0

            return {
                "total_bets": len(bets),
                "settled_bets": settled_count,
                "won_bets": won_count,
                "lost_bets": lost_count,
                "pending_bets": pending_count,
                "void_bets": void_count,
                "dry_run_bets": dry_run_count,
                "win_rate_pct": win_rate_pct,
                "total_staked_ngn": round(total_staked, 2),
                "total_payout_ngn": round(total_payout, 2),
                "net_profit_ngn": net_profit,
                "roi_pct": roi_pct,
                "market_breakdown": market_stats
            }
        finally:
            conn.close()


# Singleton accessor
_tracker_instance = None


def get_bet_tracker() -> BetTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = BetTracker()
    return _tracker_instance
