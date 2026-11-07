#!/usr/bin/env python3
"""
Multi-Sport Predictor V4 - Main Orchestrator

This module orchestrates the complete pipeline:
1. Scraping (SportyBet codes, multi-site data)
2. Data organization (JSON with documentation)
3. AI Analysis (predictions, value detection)
4. Output (reports, recommendations)

Usage:
    python -m src.main_orchestrator --code ABC123
    python -m src.main_orchestrator --directory ./input_codes
    python -m src.main_orchestrator --online
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_extractor.sportybet_scraper import SportyBetScraper, extract_sporty_code
from data_extractor.multi_site_scraper import MultiSiteScraper, extract_from_all_sites
from data_extractor.code_reader import extract_betting_code, read_sporty_code
from models.predictor import predict_match, get_predictor
from utils.json_exporter import JSONExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration for the prediction pipeline"""
    def __init__(self):
        self.output_dir = Path("output")
        self.data_dir = Path("data")
        self.scraper_timeout = 60000
        self.max_retries = 3
        self.min_value_threshold = 0.08
        self.scrape_from_all_sites = False

    def ensure_directories(self):
        """Create output directories"""
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)


class PredictionPipeline:
    """
    Main orchestration class for the sports prediction pipeline
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.ensure_directories()
        self.exporter = JSONExporter(self.config.output_dir)

    async def process_sporty_code(self, code: str) -> Dict[str, Any]:
        """
        Process a single SportyBet booking code

        Pipeline steps:
        1. Extract matches from code
        2. Build features for each match
        3. Run AI predictions
        4. Export results to JSON

        Args:
            code: SportyBet booking code

        Returns:
            Complete results dictionary
        """
        logger.info(f"Processing SportyBet code: {code}")

        # Step 1: Extract matches
        extraction_start = datetime.now(timezone.utc)
        try:
            matches = await extract_sporty_code(code)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            matches = []

        if not matches:
            return {
                "status": "error",
                "code": code,
                "message": "No matches extracted",
                "timestamp": extraction_start.isoformat()
            }

        logger.info(f"Extracted {len(matches)} matches")

        # Step 2 & 3: Analyze each match
        predictions = []
        for match in matches:
            try:
                prediction = await self._analyze_match(match)
                predictions.append({
                    "match": match,
                    "prediction": prediction,
                    "analysis_time": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                logger.error(f"Analysis failed for {match.get('home_team', 'Unknown')}: {e}")
                predictions.append({
                    "match": match,
                    "error": str(e)
                })

        # Step 4: Export to JSON
        extraction_end = datetime.now(timezone.utc)

        results = {
            "metadata": {
                "code": code,
                "status": "success",
                "scraping_time": (extraction_end - extraction_start).total_seconds(),
                "matches_extracted": len(matches),
                "matches_analyzed": len(predictions),
                "extracted_at": extraction_start.isoformat(),
                "processed_at": extraction_end.isoformat()
            },
            "matches": matches,
            "predictions": predictions,
            "value_bets": self._extract_value_bets(predictions)
        }

        # Save JSON
        output_file = self.config.output_dir / f"code_{code}_{extraction_end.strftime('%Y%m%d_%H%M%S')}.json"
        await self.exporter.export_to_json(results, str(output_file))

        return results

    async def _analyze_match(self, match: Dict) -> Dict[str, Any]:
        """Run AI analysis on a single match"""
        features = {
            "home_team": match.get("home_team", "Unknown"),
            "away_team": match.get("away_team", "Unknown"),
            "home_xG": match.get("home_xG", 1.5),
            "away_xG": match.get("away_xG", 1.5),
            "home_xGA": match.get("home_xGA", 1.2),
            "away_xGA": match.get("away_xGA", 1.2),
            "home_ppda": match.get("home_ppda", 10.5),
            "away_ppda": match.get("away_ppda", 10.5),
            "home_injury": match.get("home_injury", 0.1),
            "away_injury": match.get("away_injury", 0.1),
            "home_fatigue": match.get("home_fatigue", 0.2),
            "away_fatigue": match.get("away_fatigue", 0.2),
            "odds_home": match.get("odds_home", 0),
            "odds_draw": match.get("odds_draw", 0),
            "odds_away": match.get("odds_away", 0),
            "odds_btts": match.get("odds_btts", 0),
            "odds_over_25": match.get("odds_over_25", 0)
        }

        prediction = predict_match(features, match.get("sport", "football"))

        # Add value detection
        result = {
            "prediction": prediction,
            "is_value_bet": self._check_value_bet(prediction),
            "recommendation": self._get_recommendation(prediction)
        }

        return result

    def _check_value_bet(self, prediction: Dict) -> bool:
        """Check if prediction contains a value bet"""
        h2h = prediction.get("h2h", {})
        value = h2h.get("value", -1)
        return value > self.config.min_value_threshold

    def _get_recommendation(self, prediction: Dict) -> str:
        """Generate betting recommendation"""
        h2h = prediction.get("h2h", {})
        value = h2h.get("value", -1)

        if value > 0.15:
            return "STRONG BUY"
        elif value > 0.08:
            return "BUY"
        elif value > 0:
            return "MONITOR"
        else:
            return "SKIP"

    def _extract_value_bets(self, predictions: List[Dict]) -> List[Dict]:
        """Extract all value bets from predictions"""
        value_bets = []
        for pred in predictions:
            if "prediction" in pred:
                if self._check_value_bet(pred["prediction"]):
                    value_bets.append({
                        "match": pred["prediction"]["h2h"]["match"],
                        "outcome": pred["prediction"]["h2h"]["outcome"],
                        "value": pred["prediction"]["h2h"]["value"],
                        "confidence": pred["prediction"]["h2h"]["model_probability"],
                        "recommendation": pred.get("recommendation", "BUY")
                    })
        return value_bets

    async def process_multiple_codes(self, codes: List[str]) -> List[Dict]:
        """Process multiple SportyBet codes in parallel"""
        tasks = [self.process_sporty_code(code) for code in codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed.append({
                    "status": "error",
                    "code": codes[i],
                    "error": str(result)
                })
            else:
                processed.append(result)

        return processed

    async def scrape_all_sites(self, code: str) -> Dict[str, Any]:
        """Scrape the same code from multiple bookmaker sites"""
        results = await extract_from_all_sites(code)

        # Aggregate results
        all_matches = []
        sources = []
        for site_result in results:
            if site_result.get("matches"):
                all_matches.extend(site_result["matches"])
                sources.append(site_result.get("source", "unknown"))

        return {
            "code": code,
            "sources_scraped": list(set(sources)),
            "total_matches": len(all_matches),
            "matches": all_matches
        }


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Sport Predictor V4 - AI-Powered Betting Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--code', '-c',
        type=str,
        help='SportyBet booking code to analyze'
    )

    parser.add_argument(
        '--codes', '-f',
        type=str,
        help='File containing multiple codes (one per line)'
    )

    parser.add_argument(
        '--directory', '-d',
        type=str,
        help='Directory containing code files'
    )

    parser.add_argument(
        '--all-sites',
        action='store_true',
        help='Scrape from all available bookmaker sites'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory for JSON results'
    )

    args = parser.parse_args()

    # Configure pipeline
    config = PipelineConfig()
    config.output_dir = Path(args.output)
    config.scrape_from_all_sites = args.all_sites
    config.ensure_directories()

    pipeline = PredictionPipeline(config)

    # Collect codes to process
    codes = []

    if args.code:
        codes = [args.code]
    elif args.codes:
        with open(args.codes, 'r') as f:
            codes = [line.strip() for line in f if line.strip()]
    elif args.directory:
        dir_path = Path(args.directory)
        for file_path in dir_path.glob("*.txt"):
            with open(file_path, 'r') as f:
                codes.extend([line.strip() for line in f if line.strip()])
    else:
        print("No codes provided. Use --code, --codes, or --directory")
        parser.print_help()
        return

    # Process codes
    if len(codes) == 1:
        result = await pipeline.process_sporty_code(codes[0])
    else:
        result = await pipeline.process_multiple_codes(codes)

    # Print summary
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")

    if isinstance(result, list):
        total = sum(1 for r in result if isinstance(r, dict) and r.get("status") == "success")
        print(f"Processed: {len(result)} codes")
        print(f"Success: {total}")
    else:
        if result.get("status") == "success":
            print(f"Matches extracted: {result['metadata']['matches_extracted']}")
            print(f"Matches analyzed: {result['metadata']['matches_analyzed']}")

            value_bets = result.get("value_bets", [])
            if value_bets:
                print(f"\nVALUE BETS DETECTED: {len(value_bets)}")
                for bet in value_bets[:5]:  # Show top 5
                    print(f"  - {bet['match']}: {bet['outcome']} (Value: {bet['value']:.3f})")

    print(f"\nResults saved to: {args.output}")

    return result


if __name__ == "__main__":
    asyncio.run(main())