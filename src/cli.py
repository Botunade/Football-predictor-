#!/usr/bin/env python3
"""
Multi-Sport Predictor V4 - CLI Entry Point
Command-line interface for sports prediction analysis
"""

import argparse
import asyncio
import json
import logging
import sys
import io
import re

# Ensure UTF-8 output and silence Windows asyncio proactor deallocator bug
if sys.platform == "win32":
    if hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
    try:
        import asyncio.proactor_events
        import asyncio.base_subprocess
        def _silence_del(self): pass
        asyncio.proactor_events._ProactorBasePipeTransport.__del__ = _silence_del
        asyncio.base_subprocess.BaseSubprocessTransport.__del__ = _silence_del
    except Exception:
        pass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_extractor import extract_sporty_code, MatchData, extract_betting_code
from data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
from feature_engineering import build_features, FeatureScaler
from models import MatchPredictor, PredictionResult
from models.ai_agent import get_ai_agent
from config import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def analyze_sporty_code(code: str) -> Dict:
    """Analyze a SportyBet booking code"""
    print(f"\n{'='*60}")
    print(f"Analyzing SportyBet Code: {code}")
    print(f"{'='*60}\n")

    # Extract matches
    print(f"[{datetime.now().isoformat()}] Extracting matches from SportyBet...")
    matches = await extract_sporty_code(code)

    if not matches:
        return {
            "status": "error",
            "message": "Failed to extract matches from code",
            "code": code
        }

    print(f"✅ Found {len(matches)} matches")

    # Initialize predictor, aggregator and AI agent
    predictor = MatchPredictor()
    aggregator = get_stats_aggregator()
    ai_agent = get_ai_agent()

    # Analyze each match
    results = []
    for i, match in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}] Analyzing: {match['home_team']} vs {match['away_team']}")

        try:
            features = await build_features(match, sport="football")
            prediction = predictor.predict_match(features.to_dict(), sport="football")
            stats_obj = await aggregator.get_match_stats(match.get("home_team", ""), match.get("away_team", ""))
            consensus_stats = stats_obj.to_dict()

            # Run AI Analytical Agent
            ai_analysis = ai_agent.analyze_fixture(match, consensus_stats, prediction)

            result = {
                "match": match,
                "prediction": prediction,
                "consensus_stats": consensus_stats,
                "ai_analysis": ai_analysis,
                "analysis": analyze_match_value(prediction)
            }
            results.append(result)

            # Print summary
            h2h = prediction["h2h"]
            print(f"    Selected Pick : {match.get('selected_outcome', 'N/A')} @ {match.get('selected_odds', 0.0):.2f}")
            print(f"    Model Outcome : {h2h['outcome']} (Confidence: {h2h['model_probability']*100:.1f}%, Fair Edge: {h2h['value']:+.3f})")

            # Scraped stats summary
            if consensus_stats.get("has_forebet"):
                fb_p = consensus_stats.get("forebet_prob", {})
                print(f"    Forebet Tip   : Pred {consensus_stats.get('forebet_score')} (Prob H:{fb_p.get('home', 0)*100:.0f}% D:{fb_p.get('draw', 0)*100:.0f}% A:{fb_p.get('away', 0)*100:.0f}%)")
            if consensus_stats.get("has_predictz"):
                print(f"    PredictZ Tip  : {consensus_stats.get('predictz_outcome')} (Score: {consensus_stats.get('predictz_score', 'N/A')})")
            if consensus_stats.get("has_matchoutlook"):
                print(f"    MatchOutlook  : Best Bet: {consensus_stats.get('matchoutlook_best_bet')} ({consensus_stats.get('matchoutlook_desc', '')})")
            if consensus_stats.get("has_scoregpt"):
                print(f"    ScoreGPT AI   : Pick: {consensus_stats.get('scoregpt_pick')} ({consensus_stats.get('scoregpt_score', '')}) - {consensus_stats.get('scoregpt_panel', '')}")
            if consensus_stats.get("has_betensured"):
                print(f"    Betensured    : Tip: {consensus_stats.get('betensured_tip')} (Score: {consensus_stats.get('betensured_score', '')})")

            # Multi-Market Betting Options
            mm = prediction.get("multi_markets", {})
            if mm:
                dc = mm.get("double_chance", {})
                tot = mm.get("totals", {})
                btts = mm.get("btts", mm.get("both_teams_to_score", {}))
                ah = mm.get("asian_handicap", {})
                top_s = mm.get("top_correct_scores", [])
                score_str = ", ".join([f"{s['score']} ({s['prob']*100:.0f}%)" for s in top_s[:2]])
                print(f"    🎲 Other Markets:")
                print(f"       * Double Chance : 1X: {dc.get('1X (Home/Draw)', {}).get('prob', 0)*100:.0f}% | X2: {dc.get('X2 (Draw/Away)', {}).get('prob', 0)*100:.0f}% | 12: {dc.get('12 (Home/Away)', {}).get('prob', 0)*100:.0f}%")
                print(f"       * Totals (Goals): Over 1.5: {tot.get('Over 1.5', {}).get('prob', 0)*100:.0f}% | Over 2.5: {tot.get('Over 2.5', {}).get('prob', 0)*100:.0f}%")
                print(f"       * BTTS (Goal-Goal): Yes {btts.get('BTTS Yes', {}).get('prob', 0)*100:.0f}% | No {btts.get('BTTS No', {}).get('prob', 0)*100:.0f}%")
                if ah:
                    print(f"       * Asian Handicap: Home +0.5: {ah.get('Home +0.5', {}).get('prob', 0)*100:.0f}% | Away +0.5: {ah.get('Away +0.5', {}).get('prob', 0)*100:.0f}%")
                if score_str:
                    print(f"       * Likely Scores : {score_str}")

            # AI Analytical Agent Verdict & Synthesis
            print(f"    🧠 AI Verdict  : {ai_analysis.get('ai_verdict')}")
            print(f"    📊 Confidence  : {ai_analysis.get('confidence_score')}/10 | Recommended Stake: {ai_analysis.get('recommended_stake')}")
            print(f"    📝 Tactical    : {ai_analysis.get('tactical_overview')}")

            # Best alternative bets suggested by AI
            alt_bets = ai_analysis.get("best_alternative_options", [])
            if alt_bets:
                alt_strs = [f"{a.get('option')} ({a.get('rationale', '')})" for a in alt_bets]
                print(f"    💡 Alternative Bet Options: {'; '.join(alt_strs)}")

            if result["analysis"]["is_value"]:
                print(f"    💰 VALUE BET DETECTED!")

        except Exception as e:
            logger.error(f"Analysis error for match {i}: {e}")
            results.append({
                "match": match,
                "error": str(e)
            })

    return {
        "status": "success",
        "code": code,
        "matches_extracted": len(matches),
        "matches_analyzed": len(results),
        "results": results,
        "analysis_time": datetime.now(timezone.utc).isoformat()
    }


def analyze_match_value(prediction: Dict) -> Dict:
    """Analyze prediction for value bet opportunities"""
    h2h = prediction.get("h2h", {})
    value = h2h.get("value", -1)
    model_prob = h2h.get("model_probability", 0)

    # Value bet is when model confidence exceeds market odds
    is_value = value > 0.08

    return {
        "is_value": is_value,
        "recommendation": "PLACE BET" if is_value else "MONITOR",
        "value_score": value,
        "model_confidence": model_prob
    }


def manual_predict(home: str, away: str, odds: List[float], sport: str = "football") -> Dict:
    """Perform manual prediction for given match and odds"""
    print(f"\n{'='*60}")
    print(f"Manual Prediction: {home} vs {away}")
    print(f"Sport: {sport.title()}")
    print(f"{'='*60}\n")

    # Build fixture mock
    fixture = {
        "home_team": home,
        "away_team": away,
        "odds_home": odds[0] if len(odds) > 0 else 2.0,
        "odds_draw": odds[1] if len(odds) > 1 else 3.0,
        "odds_away": odds[2] if len(odds) > 2 else 4.0
    }

    # Build features
    features = asyncio.run(build_features(fixture, sport))

    # Predict
    predictor = MatchPredictor()
    prediction = predictor.predict_match(features.to_dict(), sport)

    # Print results
    h2h = prediction["h2h"]
    print(f"🏆 Outcome: {h2h['outcome']}")
    print(f"📈 Model Confidence: {h2h['model_probability']*100:.1f}%")
    print(f"💰 Implied Probability: {h2h['implied_probability']*100:.1f}%")
    print(f"📊 Value Score: {h2h['value']:.4f}")

    if h2h['value'] > 0.08:
        print("\n💰 VALUE BET RECOMMENDATION: Consider placing this bet!")

    return prediction


def train_model(data_file: str, output_dir: str = ".") -> Dict:
    """Train a model from historical data file"""
    import pandas as pd

    print(f"\nTraining model from: {data_file}")

    # Load data
    try:
        if data_file.endswith('.csv'):
            df = pd.read_csv(data_file)
        else:
            with open(data_file, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load data: {e}"}

    # Convert to feature format
    matches = df.to_dict('records')

    trainer = ModelTrainer(
        model_path=f"{output_dir}/model.pkl",
        scaler_path=f"{output_dir}/scaler.pkl"
    )

    metrics = trainer.train(matches)

    print(f"\nTraining Results:")
    print(f"  Samples: {metrics.get('sample_count', 0)}")
    if 'h2h' in metrics:
        print(f"  H2H Accuracy: {metrics['h2h'].get('accuracy', 0):.2%}")

    return {
        "status": "success",
        "metrics": metrics,
        "model_path": f"{output_dir}/model.pkl"
    }


def read_sporty_code_cli(code: str, output_file: str, headless: bool) -> Dict:
    """
    Extract SportyBet code data using the code reader module.

    This function provides the full extraction with JSON output,
    including match details, odds, metadata, and error tracking.
    """
    print(f"\n{'='*60}")
    print(f"Reading SportyBet Code: {code}")
    print(f"{'='*60}\n")

    print(f"[{datetime.now().isoformat()}] Extracting matches from SportyBet...")

    result = extract_betting_code(
        code=code,
        headless=headless,
        output_file=output_file,
        use_api_fallback=True,
        timeout=30000,
        max_retries=3
    )

    extraction = result.get("extraction_details", {})
    matches = result.get("matches", [])
    metadata = result.get("metadata", {})

    if extraction.get("status") == "success":
        print(f"✅ Successfully extracted {len(matches)} matches")
        print(f"   Method: {metadata.get('method', 'unknown')}")
    else:
        print(f"❌ Extraction failed: {extraction.get('error_message', 'Unknown error')}")

    if matches:
        print(f"\nExtracted Matches:")
        for i, match in enumerate(matches, 1):
            odds = match.get('odds', {})
            if isinstance(odds, dict):
                h_d_a = odds.get('h_d_a', odds)
                print(f"  {i}. {match.get('home_team', 'N/A')} vs {match.get('away_team', 'N/A')}")
                print(f"     Odds: H={h_d_a.get('home', 0):.2f} D={h_d_a.get('draw', 0):.2f} A={h_d_a.get('away', 0):.2f}")

    return result


def main():
    """CLI Main Entry Point"""
    parser = argparse.ArgumentParser(
        description="Multi-Sport Predictor V4 - AI-Powered Betting Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --code ABC123XYZ              Analyze SportyBet code
  %(prog)s --manual "Arsenal" "Chelsea" 2.1 3.5 4.2
  %(prog)s --train historical_data.json
  %(prog)s --predict-match football "Arsenal" "Chelsea" 2.1 3.5 4.2
        """
    )

    parser.add_argument(
        '--code', '-c',
        type=str,
        help='SportyBet booking code to analyze'
    )

    parser.add_argument(
        '--manual', '-m',
        nargs='+',
        help='Manual prediction: home_team away_team odds_h [odds_d] [odds_a] [sport]'
    )

    parser.add_argument(
        '--train', '-t',
        type=str,
        help='Train model from historical data file (JSON or CSV)'
    )

    parser.add_argument(
        '--predict-match',
        nargs=3,
        help='Quick prediction: sport home_team away_team'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='prediction_results.json',
        help='Output file for results'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--read-code', '-r',
        type=str,
        help='Read SportyBet booking code (extract matches and odds to JSON)'
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        default=True,
        help='Run browser in headless mode (default: True)'
    )

    parser.add_argument(
        '--show-browser',
        action='store_true',
        help='Show browser window (for debugging)'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Configure from environment
    load_config()

    # Determine output file
    output_file = args.output

    # Dispatch commands
    if args.code:
        result = asyncio.run(analyze_sporty_code(args.code))
    elif args.read_code:
        # Use the code reader module for full extraction
        result = read_sporty_code_cli(
            code=args.read_code,
            output_file=output_file,
            headless=(args.headless and not args.show_browser)
        )
    elif args.manual:
        if len(args.manual) < 3:
            print("Error: Manual prediction requires: home_team away_team odds_h [odds_d] [odds_a]")
            sys.exit(1)

        home = args.manual[0]
        away = args.manual[1]
        odds = [float(x) for x in args.manual[2:]]
        sport = args.manual[5] if len(args.manual) > 5 else "football"

        result = manual_predict(home, away, odds, sport)
    elif args.train:
        result = train_model(args.train)
    elif args.predict_match:
        sport, home, away = args.predict_match
        # Would need odds input
        print(f"Quick prediction for {home} vs {away} ({sport})")
        result = {"status": "quick_predict", "sport": sport, "teams": f"{home} vs {away}"}
    else:
        parser.print_help()
        sys.exit(0)

    # Output results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print("Done!")


if __name__ == "__main__":
    main()