"""
Multi-Sport Predictor V4 Web Interface
Simple web app for interacting with the prediction pipeline
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Enforce Windows Proactor Loop & Silence Transport Deallocator Bug
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.proactor_helper import patch_windows_asyncio
    patch_windows_asyncio()
except Exception:
    pass

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from data_extractor.code_reader import extract_betting_code, extract_betting_code_async
from data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
from feature_engineering import build_features
from models.predictor import predict_match
from models.ai_agent import get_ai_agent
from automation.automation_manager import get_automation_manager
from automation.bet_tracker import get_bet_tracker
from automation.scheduler import get_autonomous_scheduler
from automation.notifier import get_notifier

app = FastAPI(title="Multi-Sport Predictor V4", description="AI-Powered Sports Betting Analysis")

# Mount static files and templates
templates_dir = Path(__file__).parent / "templates"
static_dir = Path(__file__).parent / "static"

# Create directories if they don't exist
templates_dir.mkdir(exist_ok=True, parents=True)
static_dir.mkdir(exist_ok=True, parents=True)

templates = Jinja2Templates(directory=str(templates_dir))

# Global storage for results
results_storage: Dict[str, Any] = {}


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
async def startup_event():
    create_templates()
    # Pre-warm statistics aggregator in background
    asyncio.create_task(get_stats_aggregator().refresh_if_needed())
    # Perform non-blocking background probe of 9Router
    async def _probe_ai():
        try:
            probe = await asyncio.to_thread(get_ai_agent().test_connection)
            if probe.get("connected"):
                logger.info(f"9Router AI Engine: ACTIVE ({probe.get('model')} - Latency: {probe.get('latency_sec')}s)")
            else:
                logger.info(f"9Router AI Engine: OFFLINE ({probe.get('message')}) - Quantitative Fallback Active")
        except Exception as e:
            logger.debug(f"AI probe notice: {e}")
    asyncio.create_task(_probe_ai())


@app.get("/api/health/ai", response_class=JSONResponse)
async def api_health_ai():
    """Returns real-time health probe for 9Router port 20128 and active AI model."""
    try:
        probe = await asyncio.to_thread(get_ai_agent().test_connection)
        return probe
    except Exception as e:
        return {"connected": False, "error": str(e), "message": "Failed to query AI engine"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with input form and live scraped fixtures catalog"""
    aggregator = get_stats_aggregator()
    scraped_fixtures = aggregator.get_all_scraped_fixtures()
    leagues = sorted(list(set(f.get("league", "Other") for f in scraped_fixtures if f.get("league"))))

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "scraped_fixtures": scraped_fixtures,
            "leagues": leagues,
            "total_scraped": len(scraped_fixtures)
        }
    )


@app.get("/api/scraped-fixtures", response_class=JSONResponse)
async def get_scraped_fixtures():
    """API endpoint to get all current scraped fixtures from the 5 platforms"""
    aggregator = get_stats_aggregator()
    fixtures = aggregator.get_all_scraped_fixtures()
    return {"status": "success", "count": len(fixtures), "fixtures": fixtures}


@app.post("/api/refresh-scraped", response_class=JSONResponse)
async def api_refresh_scraped():
    """API endpoint to trigger immediate refresh of all 5 prediction sources"""
    try:
        aggregator = get_stats_aggregator()
        await aggregator.refresh_if_needed(force=True)
        fixtures = aggregator.get_all_scraped_fixtures()
        return {
            "status": "success",
            "message": f"Successfully refreshed {len(fixtures)} live fixtures across 5 sources",
            "count": len(fixtures),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/sportybet/sync-odds", response_class=JSONResponse)
async def api_sync_sportybet_odds(home: str, away: str):
    """
    On-demand live SportyBet odds synchronization for any fixture.
    Resolves real-time 1X2 odds, multi-market odds, and official SportyBet eventId.
    """
    try:
        aggregator = get_stats_aggregator()
        sb_match = aggregator.resolve_match_sportybet_odds(home, away)
        if sb_match and sb_match.odds_home > 1.0 and sb_match.odds_away > 1.0:
            return {
                "status": "success",
                "found": True,
                "home_team": sb_match.home_team,
                "away_team": sb_match.away_team,
                "odds_home": sb_match.odds_home,
                "odds_draw": sb_match.odds_draw,
                "odds_away": sb_match.odds_away,
                "sporty_id": sb_match.sporty_id,
                "market_odds": sb_match.market_odds or {},
                "league": sb_match.league
            }
        else:
            return {
                "status": "success",
                "found": False,
                "message": f"No active SportyBet market currently open for {home} vs {away}"
            }
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/preview-code")
async def preview_code(request: Request):
    """Preview matches in a SportyBet booking code without running full analysis"""
    try:
        code = ""
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.json()
                code = str(body.get("code", "")).strip()
            except Exception:
                code = ""
        else:
            try:
                form = await request.form()
                code = str(form.get("code", "")).strip()
            except Exception:
                code = ""

        if not code:
            return JSONResponse({"status": "error", "message": "Code is required"}, status_code=400)

        result = await extract_betting_code_async(code=code, headless=True, max_retries=1)
        matches = result.get("matches", [])
        return JSONResponse({
            "status": "success",
            "code": code,
            "match_count": len(matches),
            "matches": matches
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/automation/place-bet")
async def api_place_bet(request: Request):
    """
    Automated SportyBet Wager API
    Allows triggering a test wager of 10 NGN (dry-run or live) on a booking code.
    """
    try:
        data = {}
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                data = await request.json()
            except Exception:
                data = {}
        else:
            try:
                form = await request.form()
                data = dict(form)
            except Exception:
                data = {}

        code = str(data.get("code", "")).strip().upper()
        if not code or len(code) > 12 or " " in code:
            return JSONResponse({
                "success": False,
                "message": f"Invalid SportyBet booking code format '{code}'. Please enter a valid SportyBet booking code (e.g. X1XUKH, LEA2BS) to generate the betslip."
            }, status_code=400)

        stake = float(data.get("stake", 10.0))
        dry_run = str(data.get("dry_run", "true")).lower() in ["true", "1", "yes"]
        confirm = str(data.get("confirm", "false")).lower() in ["true", "1", "yes"]

        mgr = get_automation_manager()
        res = await mgr.execute_booking_code_wager(
            code=code,
            stake=stake,
            dry_run=dry_run,
            confirm=confirm,
            headless=True
        )

        # Log into BetTracker journal on success
        if res.get("success"):
            try:
                tracker = get_bet_tracker()
                tracker.record_bet(
                    booking_code=code,
                    stake=stake,
                    combined_odds=float(res.get("odds") or 2.0),
                    legs=[],
                    ticket_id=str(res.get("ticket_id") or ""),
                    is_dry_run=dry_run,
                    notes=f"Auto-bet from Web UI (dry_run={dry_run})"
                )
            except Exception as e_track:
                logger.debug(f"Tracker logging notice: {e_track}")

        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.get("/api/tracker/bets", response_class=JSONResponse)
async def api_tracker_get_bets(limit: int = 50, include_dry_runs: bool = True):
    """Retrieve recorded bets with leg details for the P&L Journal"""
    try:
        tracker = get_bet_tracker()
        return {"status": "success", "bets": tracker.get_bets(limit=limit, include_dry_runs=include_dry_runs)}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/tracker/summary", response_class=JSONResponse)
async def api_tracker_get_summary(include_dry_runs: bool = False):
    """Retrieve P&L summary, ROI, and market-by-market strike rate analytics"""
    try:
        tracker = get_bet_tracker()
        return {"status": "success", "summary": tracker.get_pnl_summary(include_dry_runs=include_dry_runs)}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.post("/api/tracker/settle", response_class=JSONResponse)
async def api_tracker_settle_bet(request: Request):
    """Manually or automatically settle a bet as WON, LOST, or VOID"""
    try:
        body = await request.json()
        bet_id = int(body.get("bet_id"))
        status = str(body.get("status", "WON")).upper()
        payout = body.get("payout")
        if payout is not None:
            payout = float(payout)

        tracker = get_bet_tracker()
        ok = tracker.settle_bet(bet_id=bet_id, status=status, actual_payout=payout)
        return {"status": "success" if ok else "not_found", "bet_id": bet_id, "settled_as": status}
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/automation/balance", response_class=JSONResponse)
async def api_get_balance(refresh: bool = False):
    """Fetch live SportyBet account balance."""
    try:
        mgr = get_automation_manager()
        res = await mgr.get_account_balance(force_refresh=refresh)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e), "balance_ngn": 70.93, "balance_text": "NGN 70.93"})


@app.post("/api/automation/generate-booking-code", response_class=JSONResponse)
async def api_generate_booking_code(request: Request):
    """Generate a real SportyBet booking code for a list of matches."""
    try:
        body = await request.json()
        matches = body.get("matches", [])
        if not matches:
            session_id = body.get("session_id", "")
            if session_id and session_id in results_storage:
                matches = results_storage[session_id].get("matches", [])

        try:
            from .automation.automation_manager import get_automation_manager
        except (ImportError, ValueError):
            from automation.automation_manager import get_automation_manager

        mgr = get_automation_manager()
        res = await mgr.generate_booking_code(matches=matches)
        return JSONResponse(res)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e), "code": "X1XUKH"})


@app.post("/api/simulate-acca", response_class=JSONResponse)
async def api_simulate_acca(request: Request):
    """Simulate optimal accumulator combination matching target odds and match count."""
    try:
        body = await request.json()
        target_odds = float(body.get("target_odds", 10.0))
        match_count = int(body.get("match_count", 5))
        strategy = str(body.get("strategy", "max_prob"))
        stake = float(body.get("stake", 10.0) or 10.0)
        market_filter = str(body.get("market_filter", "all"))
        variation_index = int(body.get("variation_index", 0) or 0)

        from data_extractor.stats_scrapers.stats_aggregator import get_stats_aggregator
        from models.acca_simulator import AccaSimulator

        aggregator = get_stats_aggregator()
        fixtures = aggregator.get_all_scraped_fixtures()
        sim = AccaSimulator(fixtures)
        result = sim.simulate_optimal_slip(
            target_odds=target_odds,
            match_count=match_count,
            strategy=strategy,
            stake=stake,
            market_filter=market_filter,
            variation_index=variation_index
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


@app.post("/analyze-selected", response_class=HTMLResponse)
async def analyze_selected_matches(request: Request, selected_fixtures_json: str = Form(...), source_title: str = Form("Custom Scraped Selection")):
    """Analyze a custom list of selected scraped games"""
    try:
        raw_matches = json.loads(selected_fixtures_json)
        if not raw_matches:
            raise ValueError("No matches were selected for analysis")

        aggregator = get_stats_aggregator()
        matches = []
        for m in raw_matches:
            h_name = str(m.get("home_team", "")).strip()
            a_name = str(m.get("away_team", "")).strip()
            sp_id = str(m.get("sporty_id") or "").strip()
            oh = float(m.get("odds_home", 2.10))
            od = float(m.get("odds_draw", 3.20))
            oa = float(m.get("odds_away", 3.10))
            m_odds = dict(m.get("market_odds") or {})

            # Auto-resolve live SportyBet ground-truth if unverified or generic fallback
            sb_m = aggregator.resolve_match_sportybet_odds(h_name, a_name)
            if sb_m and sb_m.odds_home > 1.0 and sb_m.odds_away > 1.0:
                sp_id = sb_m.sporty_id or sp_id
                # If user had generic default odds, update with live SportyBet true odds
                if (oh == 2.10 and od == 3.20 and oa == 3.10) or not m.get("odds_verified"):
                    oh = float(sb_m.odds_home)
                    od = float(sb_m.odds_draw) if sb_m.odds_draw > 1.0 else 3.20
                    oa = float(sb_m.odds_away)
                if sb_m.market_odds:
                    m_odds.update(sb_m.market_odds)

            selected_out = m.get("selected_outcome", "Home")
            # If user had generic default pick and real odds inverted, align pick
            if selected_out == "Home" and (oh == 2.10 and od == 3.20 and oa == 3.10) and sb_m:
                selected_out = "Away" if oa < oh else "Home"

            sel_odds = float(m.get("selected_odds", 0.0))
            if sel_odds <= 1.0:
                sel_odds = m_odds.get(selected_out, oh if selected_out == "Home" else (oa if selected_out == "Away" else od))

            matches.append({
                "home_team": h_name,
                "away_team": a_name,
                "league": m.get("league", "General"),
                "odds_home": oh,
                "odds_draw": od,
                "odds_away": oa,
                "selected_outcome": selected_out,
                "selected_odds": sel_odds,
                "sporty_id": sp_id,
                "sporty_market_id": m.get("sporty_market_id", ""),
                "sporty_outcome_id": m.get("sporty_outcome_id", ""),
                "sporty_specifier": m.get("sporty_specifier", ""),
                "market_odds": m_odds,
                "sport": "football",
                "status": "upcoming"
            })

        ai_agent = get_ai_agent()
        sem = asyncio.Semaphore(2)

        async def process_single_match(match):
            try:
                feat = await build_features(match, "football")
                feat_dict = feat.to_dict()
                feat_dict["selected_outcome"] = match.get("selected_outcome", "")
                feat_dict["selected_odds"] = match.get("selected_odds", 0.0)
                feat_dict["market_odds"] = match.get("market_odds", {})
                prediction = predict_match(feat_dict, "football")
                stats_obj = await aggregator.get_match_stats(match.get("home_team", ""), match.get("away_team", ""))
                c_stats = stats_obj.to_dict()
                c_stats["h2h_data"] = feat.h2h_data
                c_stats["league_context"] = feat.league_context
                c_stats["sharp_money"] = feat.sharp_money

                async with sem:
                    ai_analysis = await asyncio.to_thread(ai_agent.analyze_fixture, match, c_stats, prediction)

                return {
                    "match": match,
                    "prediction": prediction,
                    "consensus_stats": c_stats,
                    "enrichment": {
                        "h2h": feat.h2h_data,
                        "league_context": feat.league_context,
                        "sharp_money": feat.sharp_money
                    },
                    "ai_analysis": ai_analysis
                }
            except Exception as e:
                return {
                    "match": match,
                    "error": str(e)
                }

        predictions = await asyncio.gather(*[process_single_match(m) for m in matches])

        value_bets = []
        for pred in predictions:
            if "prediction" in pred:
                h2h = pred["prediction"].get("h2h", {})
                value = h2h.get("value", -1)
                if value > 0.08:
                    value_bets.append({
                        "match": h2h.get("match"),
                        "outcome": h2h.get("outcome"),
                        "value": value,
                        "confidence": h2h.get("model_probability")
                    })

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "code": source_title,
            "metadata": {"code": source_title, "method": "Selected Scraped Fixtures"},
            "extraction_details": {"status": "success", "selection_type": "user_selected"},
            "matches": matches,
            "predictions": predictions,
            "value_bets": value_bets,
            "analysis_time": datetime.now().isoformat()
        }
        results_storage[session_id] = result_data

        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "request": request,
                "result": result_data,
                "session_id": session_id
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "request": request,
                "error": str(e),
                "error_type": type(e).__name__
            },
            status_code=500
        )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_code(request: Request, code: str = Form(...)):
    """Analyze a SportyBet booking code"""
    try:
        # Extract matches from code asynchronously
        result = await extract_betting_code_async(code=code, headless=True, max_retries=1)

        matches = result.get("matches", [])
        metadata = result.get("metadata", {})
        extraction_details = result.get("extraction_details", {})

        # Run predictions and AI analysis on each match concurrently with controlled concurrency (Semaphore=2)
        aggregator = get_stats_aggregator()
        ai_agent = get_ai_agent()
        sem = asyncio.Semaphore(2)

        async def process_single_match(match):
            try:
                feat = await build_features(match, "football")
                feat_dict = feat.to_dict()
                feat_dict["selected_outcome"] = match.get("selected_outcome", "")
                feat_dict["selected_odds"] = match.get("selected_odds", 0.0)
                feat_dict["market_odds"] = match.get("market_odds", {})
                prediction = predict_match(feat_dict, "football")
                stats_obj = await aggregator.get_match_stats(match.get("home_team", ""), match.get("away_team", ""))
                c_stats = stats_obj.to_dict()
                c_stats["h2h_data"] = feat.h2h_data
                c_stats["league_context"] = feat.league_context
                c_stats["sharp_money"] = feat.sharp_money

                async with sem:
                    ai_analysis = await asyncio.to_thread(ai_agent.analyze_fixture, match, c_stats, prediction)

                return {
                    "match": match,
                    "prediction": prediction,
                    "consensus_stats": c_stats,
                    "enrichment": {
                        "h2h": feat.h2h_data,
                        "league_context": feat.league_context,
                        "sharp_money": feat.sharp_money
                    },
                    "ai_analysis": ai_analysis
                }
            except Exception as e:
                return {
                    "match": match,
                    "error": str(e)
                }

        predictions = await asyncio.gather(*[process_single_match(m) for m in matches])

        # Find value bets
        value_bets = []
        for pred in predictions:
            if "prediction" in pred:
                h2h = pred["prediction"].get("h2h", {})
                value = h2h.get("value", -1)
                if value > 0.08:
                    value_bets.append({
                        "match": h2h.get("match"),
                        "outcome": h2h.get("outcome"),
                        "value": value,
                        "confidence": h2h.get("model_probability")
                    })

        # Store result
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_data = {
            "code": code,
            "metadata": metadata,
            "extraction_details": extraction_details,
            "matches": matches,
            "predictions": predictions,
            "value_bets": value_bets,
            "analysis_time": datetime.now().isoformat()
        }
        results_storage[session_id] = result_data

        return templates.TemplateResponse(
            request=request,
            name="results.html",
            context={
                "request": request,
                "result": result_data,
                "session_id": session_id
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="error.html",
            context={
                "request": request,
                "error": str(e),
                "error_type": type(e).__name__
            },
            status_code=500
        )


@app.get("/api/result/{session_id}", response_class=JSONResponse)
async def get_result(session_id: str):
    """Get analysis result by session ID"""
    if session_id in results_storage:
        return results_storage[session_id]
    return {"error": "Result not found"}, 404


@app.get("/results", response_class=HTMLResponse)
async def results_list(request: Request):
    """List all stored results"""
    return templates.TemplateResponse(
        request=request,
        name="results_list.html",
        context={
            "request": request,
            "results": list(results_storage.items())
        }
    )


# Create HTML templates
def create_templates():
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Sport Predictor V4</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; }
        .btn-primary:hover { background: linear-gradient(135deg, #764ba2 0%, #667eea 100%); }
        .code-input { font-size: 1.5rem; letter-spacing: 2px; }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10">
                <div class="card">
                    <div class="card-header bg-primary text-white text-center py-4">
                        <h2 class="mb-0">SportyBet Predictor V4</h2>
                        <p class="mb-0">AI-Powered Sports Betting Analysis</p>
                    </div>
                    <div class="card-body p-4">
                        <form action="/analyze" method="post">
                            <div class="mb-4">
                                <label for="code" class="form-label fw-bold">SportyBet Booking Code</label>
                                <input type="text" class="form-control code-input text-center"
                                       id="code" name="code" placeholder="e.g., ABC123XYZ"
                                       required maxlength="50">
                            </div>
                            <div class="d-grid gap-2">
                                <button type="submit" class="btn btn-primary btn-lg">Analyze Code</button>
                            </div>
                        </form>

                        <hr class="my-4">

                        <h5 class="mb-3">Features</h5>
                        <ul class="list-unstyled">
                            <li class="mb-2"><span class="text-success">+</span> Complete match extraction from SportyBet codes</li>
                            <li class="mb-2"><span class="text-success">+</span> AI-powered predictions with XGBoost models</li>
                            <li class="mb-2"><span class="text-success">+</span> Value bet detection (>8 percent threshold)</li>
                            <li class="mb-2"><span class="text-success">+</span> JSON export with documentation</li>
                            <li class="mb-2"><span class="text-success">+</span> No Telegram, no OddsAPI - clean V4</li>
                        </ul>
                    </div>
                    <div class="card-footer text-center text-muted">
                        Multi-Sport Predictor V4 - AI-Powered Betting Analysis
                    </div>
                </div>
            </div>
        </div>
    </div>
</html>"""

    results_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - Multi-Sport Predictor V4</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f5f5f5; }
        .card { border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .value-badge { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); }
    </style>
</head>
<body>
    <div class="container py-5">
        <h1 class="mb-4">Analysis Results</h1>

        <div class="row mb-4">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5 class="mb-0">Metadata</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Code:</strong> {{ result.metadata.code }}</p>
                        <p><strong>Extraction Method:</strong> {{ result.metadata.method }}</p>
                        <p><strong>Status:</strong>
                            {% if result.extraction_details.status == 'success' %}
                                <span class="text-success">{{ result.extraction_details.status }}</span>
                            {% else %}
                                <span class="text-danger">{{ result.extraction_details.status }}</span>
                            {% endif %}
                        </p>
                        <p><strong>Matches Found:</strong> {{ result.matches|length }}</p>
                        <p><strong>Predictions:</strong> {{ result.predictions|length }}</p>
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0">Quick Stats</h5>
                    </div>
                    <div class="card-body text-center">
                        {% if result.value_bets %}
                            <div class="display-4 text-success mb-2">{{ result.value_bets|length }}</div>
                            <p class="text-muted">Value Bets Detected</p>
                        {% else %}
                            <div class="display-4 text-muted mb-2">0</div>
                            <p class="text-muted">Value Bets Detected</p>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <h3 class="mb-3">Matches</h3>
        <div class="row">
            {% for match in result.matches %}
            <div class="col-md-6 mb-3">
                <div class="card match-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ match.home_team }} vs {{ match.away_team }}</h5>
                        <p class="card-text text-muted">
                            {% if match.odds %}
                                Odds: {{ match.odds.h_d_a.home if match.odds.h_d_a else match.odds_home }} /
                                {{ match.odds.draw }} / {{ match.odds.away }}
                            {% else %}
                                Odds data not available
                            {% endif %}
                        </p>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <a href="/" class="btn btn-outline-primary">Analyze Another Code</a>
    </div>
</body>
</html>"""

    error_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Multi-Sport Predictor V4</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .error-card { background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-6 col-md-8">
                <div class="error-card text-center p-4">
                    <div class="display-1 mb-3">!</div>
                    <h3 class="mb-3">Analysis Error</h3>
                    <p class="text-muted">{{ error_type }}: {{ error }}</p>
                    <a href="/" class="btn btn-primary mt-3">Try Another Code</a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    results_list_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Results - Multi-Sport Predictor V4</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container py-5">
        <h1 class="mb-4">Analysis History</h1>

        {% if results %}
            <div class="row">
                {% for session_id, result in results %}
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">Session: {{ session_id }}</h5>
                            <p class="card-text text-muted">
                                Code: {{ result.code }}<br>
                                Matches: {{ result.matches|length }}<br>
                                Time: {{ result.analysis_time }}
                            </p>
                        </div>
                    </div>
                </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="text-center py-5">
                <p class="text-muted">No results yet. Analyze a code to see results here.</p>
            </div>
        {% endif %}

        <a href="/" class="btn btn-outline-primary">Back to Analysis</a>
    </div>
</body>
</html>"""

    # Write templates if they don't already exist
    if not (templates_dir / "index.html").exists():
        (templates_dir / "index.html").write_text(index_html, encoding='utf-8')
    if not (templates_dir / "results.html").exists():
        (templates_dir / "results.html").write_text(results_html, encoding='utf-8')
    if not (templates_dir / "error.html").exists():
        (templates_dir / "error.html").write_text(error_html, encoding='utf-8')
    if not (templates_dir / "results_list.html").exists():
        (templates_dir / "results_list.html").write_text(results_list_html, encoding='utf-8')


if __name__ == "__main__":
    create_templates()
    uvicorn.run(app, host="127.0.0.1", port=8000)