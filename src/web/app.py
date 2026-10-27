"""
Multi-Sport Predictor V4 Web Interface
Simple web app for interacting with the prediction pipeline
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_extractor.code_reader import extract_betting_code
from models.predictor import predict_match

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


@app.on_event("startup")
async def startup_event():
    create_templates()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with input form"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_code(code: str = Form(...)):
    """Analyze a SportyBet booking code"""
    try:
        # Extract matches from code
        result = extract_betting_code(code=code, headless=True, max_retries=1)

        matches = result.get("matches", [])
        metadata = result.get("metadata", {})
        extraction_details = result.get("extraction_details", {})

        # Run predictions on each match
        predictions = []
        for match in matches:
            try:
                # Build features from match data
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
                    "odds_home": match.get("odds_home", 2.0),
                    "odds_draw": match.get("odds_draw", 3.0),
                    "odds_away": match.get("odds_away", 4.0),
                }

                prediction = predict_match(features, "football")
                predictions.append({
                    "match": match,
                    "prediction": prediction
                })
            except Exception as e:
                predictions.append({
                    "match": match,
                    "error": str(e)
                })

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

        return templates.TemplateResponse("results.html", {
            "request": Request,
            "result": result_data,
            "session_id": session_id
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": Request,
            "error": str(e),
            "error_type": type(e).__name__
        }, status_code=500)


@app.get("/api/result/{session_id}", response_class=JSONResponse)
async def get_result(session_id: str):
    """Get analysis result by session ID"""
    if session_id in results_storage:
        return results_storage[session_id]
    return {"error": "Result not found"}, 404


@app.get("/results", response_class=HTMLResponse)
async def results_list(request: Request):
    """List all stored results"""
    return templates.TemplateResponse("results_list.html", {
        "request": request,
        "results": list(results_storage.items())
    })


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

    # Write templates
    (templates_dir / "index.html").write_text(index_html, encoding='utf-8')
    (templates_dir / "results.html").write_text(results_html, encoding='utf-8')
    (templates_dir / "error.html").write_text(error_html, encoding='utf-8')
    (templates_dir / "results_list.html").write_text(results_list_html, encoding='utf-8')


if __name__ == "__main__":
    create_templates()
    uvicorn.run(app, host="127.0.0.1", port=8000)