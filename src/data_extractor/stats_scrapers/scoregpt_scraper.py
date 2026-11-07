"""
ScoreGPT Scraper Module
Extracts multi-model AI consensus predictions (Claude, GPT, Grok panel),
predicted scorelines, and agreement ratios from ScoreGPT (https://scoregpt.app)
"""

import asyncio
import logging
import re
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ScoreGPTPrediction:
    """Structured multi-model AI prediction from ScoreGPT"""
    home_team: str
    away_team: str
    league: str
    ai_pick: str
    predicted_score: str
    consensus_ratio: str  # e.g. "5/5 models" or "4/5 models"
    panel_verdict: str    # e.g. "Consensus back Home" or "AI panel split"
    scraped_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ScoreGPTScraper:
    """
    HTTP and BeautifulSoup scraper for ScoreGPT matchday predictions.
    Fetches multi-model AI consensus for football fixtures.
    """

    TARGET_URL = "https://scoregpt.app/ai-football-predictions/matchday"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def scrape_predictions(self) -> List[ScoreGPTPrediction]:
        """Fetch and parse all AI match predictions from ScoreGPT"""
        return await asyncio.to_thread(self._scrape_sync)

    def _scrape_sync(self) -> List[ScoreGPTPrediction]:
        predictions: List[ScoreGPTPrediction] = []
        try:
            req = urllib.request.Request(
                self.TARGET_URL,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
                }
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                html = resp.read().decode("utf-8", errors="ignore")

            soup = BeautifulSoup(html, "html.parser")
            now_str = datetime.now(timezone.utc).isoformat()

            seen_fixtures = set()

            for a in soup.find_all("a"):
                href = a.get("href", "")
                if "/ai-football-predictions/" in href and "-vs-" in href:
                    slug = href.rstrip("/").split("/")[-1]
                    # e.g. "galatasaray-vs-vfb-stuttgart-2026-11-03"
                    slug_clean = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', slug)
                    parts = slug_clean.split("-vs-")
                    if len(parts) != 2:
                        continue

                    home = parts[0].replace("-", " ").title()
                    away = parts[1].replace("-", " ").title()
                    fix_key = f"{home.lower()}_vs_{away.lower()}"

                    text = a.get_text(" ", strip=True)
                    if fix_key in seen_fixtures and ("AI consensus" in text or "AI panel" in text):
                        # Update existing
                        continue

                    ai_pick = ""
                    pred_score = ""
                    consensus = ""
                    panel_verdict = ""

                    # Parse text: "AI pick: Galatasaray - predicted 2-1, 5 of 5 models back Galatasaray"
                    pick_m = re.search(r'AI (?:pick|call):\s*([A-Za-z0-9\s\.\/]+?)(?:\s*[–\-\·\•]|\s+predicted|\.|$)', text)
                    if pick_m:
                        ai_pick = pick_m.group(1).strip()

                    score_m = re.search(r'predicted\s*(\d+[\-–]\d+)', text)
                    if score_m:
                        pred_score = score_m.group(1).replace("–", "-").strip()

                    ratio_m = re.search(r'(\d+\s*of\s*\d+\s*models|\d+/\d+)', text)
                    if ratio_m:
                        consensus = ratio_m.group(1).strip()

                    if "panel split" in text.lower():
                        panel_verdict = "Panel Split"
                    elif "consensus" in text.lower():
                        panel_verdict = f"Consensus ({consensus})" if consensus else "Strong Consensus"

                    pred = ScoreGPTPrediction(
                        home_team=home,
                        away_team=away,
                        league="Matchday",
                        ai_pick=ai_pick,
                        predicted_score=pred_score,
                        consensus_ratio=consensus,
                        panel_verdict=panel_verdict or "Multi-Model AI",
                        scraped_at=now_str
                    )
                    predictions.append(pred)
                    seen_fixtures.add(fix_key)

            logger.info(f"ScoreGPT scraped {len(predictions)} AI match consensus records")
        except Exception as e:
            logger.warning(f"Error scraping ScoreGPT: {e}")

        return predictions
