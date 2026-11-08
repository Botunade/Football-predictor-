#!/usr/bin/env python3
"""
Multi-Sport Predictor V4 - Unified Main Entrypoint
Pure Web Scraping & AI-Powered Betting Analysis Pipeline

Usage:
    python main.py --code RN4YRW
    python main.py --web
    python main.py --stats
    python main.py --train
"""

import argparse
import sys
from pathlib import Path

# Silence Windows asyncio proactor closed pipe deallocator bug & set Proactor loop
if sys.platform == "win32":
    try:
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        import asyncio.proactor_events
        import asyncio.base_subprocess
        def _silence_del(self): pass
        asyncio.proactor_events._ProactorBasePipeTransport.__del__ = _silence_del
        asyncio.base_subprocess.BaseSubprocessTransport.__del__ = _silence_del
    except Exception:
        pass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Sport Predictor V4: Pure Web Scraping & AI Betting Analysis"
    )
    parser.add_argument("--code", type=str, help="SportyBet booking code / shareCode to analyze (e.g. RN4YRW)")
    parser.add_argument("--web", action="store_true", help="Launch the FastAPI web interface at http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")
    parser.add_argument("--stats", action="store_true", help="Scrape and show latest predictions from Forebet & PredictZ")
    parser.add_argument("--train", action="store_true", help="Retrain ML models")
    parser.add_argument("--verify", action="store_true", help="Verify AI credentials and environment settings")

    args = parser.parse_args()

    if args.verify:
        from verify_creds import verify_creds
        verify_creds()
        return

    if args.web:
        import uvicorn
        print(f"Starting Multi-Sport Predictor V4 Web Interface at http://127.0.0.1:{args.port}...")
        uvicorn.run("src.web.app:app", host="127.0.0.1", port=args.port, reload=False)
        return

    if args.stats:
        from src.data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
        agg = get_stats_aggregator()
        stats = agg.refresh_all_stats()
        print(f"\n[✓] Successfully scraped {len(stats.get('forebet', []))} Forebet & {len(stats.get('predictz', []))} PredictZ matches!")
        return

    if args.train:
        from src.models.model_trainer import train_model
        train_model()
        return

    if args.code:
        import subprocess
        subprocess.run([sys.executable, "src/cli.py", "--code", args.code])
        return

    # If no arguments provided, show help
    parser.print_help()


if __name__ == "__main__":
    main()
