# Multi-Sport Predictor V4 - AI-Powered Betting Analysis

A modern, modular sports prediction system with enhanced machine learning and AI capabilities.

## Web Interface

A local web interface is available for interacting with the prediction pipeline:

```bash
# Install web dependencies
pip install -r requirements_web.txt

# Start the server
web.bat         # Windows
# or
uvicorn src.web.app:app --host 127.0.0.1 --port 8000

# Open in browser
http://127.0.0.1:8000
```

See `docs/WEB_INTERFACE.md` for full documentation.

## What Changed in V4

## 📋 What Changed in V4

- **Removed Telegram dependency** - Now uses CLI interface
- **Removed OddsAPI** - Using extracted odds only
- **Enhanced ML models** - XGBoost with calibration and ensemble methods
- **Modular architecture** - Clean separation of concerns
- **AI-enhanced predictions** - Neural networks and uncertainty quantification

## 🏗️ Project Structure

```
football_projects/
├── src/                          # New modular codebase
│   ├── data_extractor/          # Web scraping (Playwright)
│   ├── feature_engineering/     # ML feature preparation
│   ├── models/                  # Prediction models (XGBoost, Ensemble)
│   ├── utils/                   # Configuration, validation
│   └── cli.py                   # Main CLI entry point
├── .env                         # API keys (API-Sports only)
├── requirements.txt             # Updated dependencies
└── README.md
```

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Playwright browsers**:
   ```bash
   playwright install chromium --with-deps
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API-Sports key
   ```

## 🚀 Usage

### Analyze a SportyBet Code
```bash
python src/cli.py --code ABC123XYZ
```

### Manual Match Prediction
```bash
# Football (3 or more odds)
python src/cli.py --manual "Arsenal" "Chelsea" 2.1 3.5 4.2

# Basketball (2 odds)
python src/cli.py --manual "Lakers" "Celtics" 1.90 2.05

# With sport specified
python src/cli.py --manual "Lakers" "Celtics" 1.90 2.05 basketball
```

### Train a Model
```bash
# From JSON file
python src/cli.py --train historical_data.json

# From CSV file
python src/cli.py --train match_data.csv
```

## 📊 CLI Commands

| Command | Example | Description |
|---------|---------|-------------|
| `--code` | `--code ABC123` | Analyze SportyBet booking code |
| `--manual` | `--manual TeamA TeamB 2.1 3.5 4.2` | Manual match prediction |
| `--train` | `--train data.json` | Train ML model |
| `--help` | `--help` | Show help |

## 🔧 Configuration (.env)

```bash
# API-Sports Key (for fixture data)
API_KEY=your_api_sports_key

# Optional: Model settings
MODEL_N_ESTIMATORS=200
MODEL_LR=0.05
MODEL_MAX_DEPTH=5

# Scraper settings
SCRAPER_HEADLESS=true
SCRAPER_TIMEOUT=60000

# Value thresholds
MIN_VALUE_THRESHOLD=0.08
```

## 🧠 ML Models

### Prediction Markets
1. **H2H** (Home/Draw/Away) - Calibrated XGBoost
2. **BTTS** (Both Teams To Score) - Gradient Boosting
3. **Over 2.5** - Gradient Boosting

### Value Calculation
```
Value = Model Probability - Implied Probability (1/odds)
```

A bet is considered "Value" when Value > 0.08 (8%)

## 📁 Data Pipeline

```
SportyBet Code → Playwright Scraper → Match Data
                                          ↓
                              Feature Builder → Feature Vector
                                          ↓
                                  ML Predictor → Probabilities
                                          ↓
                                    Value Calculator → Bets
```

## 🧪 Testing

Run the existing tests:
```bash
python test_predictor.py
python test_sync_logic.py
```

New tests in `/tests/` directory will be added.

## 🔒 Security

- No Telegram bot (removed attack surface)
- API keys only used for:
  - API-Sports (fixture/statistics data)
  - NOT used for OddsAPI (removed in V4)
- Credentials stored in `.env` (never committed)

## 📈 Performance

- Async Playwright for non-blocking scraping
- XGBoost models for fast predictions
- Feature caching for repeated predictions
- Concurrent scraping with semaphores

## ⚠️ Migration Notes

If you're upgrading from V3:

1. Your `.env` only needs `API_KEY` now (removed `ODDS_API_KEY`, `TELEGRAM_TOKEN`, `CHAT_ID`)
2. Use CLI instead of Telegram bot
3. Old tests may need updates for deterministic model behavior
4. Models are now saved as `model_v4.pkl` and `scaler_v4.pkl`

## 🤝 Contributing

- See `src/data_extractor/` for scraping improvements
- See `src/models/` for prediction enhancements  
- See `src/feature_engineering/` for feature additions

## 📄 License

Proprietary - for internal use only