# Web Interface for Multi-Sport Predictor V4

A simple, local web interface for interacting with the sports prediction pipeline.

## Quick Start

### Install web dependencies
```bash
pip install -r requirements_web.txt
```

### Start the server
```bash
# Windows
web.bat

# Or directly
python -c "from src.web.app import app, create_templates; create_templates()"
uvicorn src.web.app:app --host 127.0.0.1 --port 8000
```

### Open in browser
Visit: http://127.0.0.1:8000

## Features

### Home Page
- Simple input form for SportyBet booking codes
- Feature summary
- Clean, modern design

### Results Page
After submitting a code:
- Match extraction results
- Prediction details
- Value bet identification
- Odds display

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page with input form |
| `/analyze` | POST | Analyze a booking code |
| `/api/result/{id}` | GET | Get JSON result by session ID |
| `/results` | GET | List all analysis sessions |

## Architecture

```
Web Interface (src/web/)
├── app.py           # FastAPI application
├── static/
│   └── style.css    # Custom styles
└── templates/
    ├── index.html   # Home page
    ├── results.html # Results page
    ├── error.html   # Error page
    └── results_list.html # Session list
```

## Security Notes

- Runs locally on 127.0.0.1 only
- No database required
- Results stored in memory (cleared on restart)
- No external API keys needed for the web interface

## Customization

Edit `src/web/app.py` to:
- Add more features
- Change styling
- Add new API endpoints