# 🚀 Multi-Sport Prediction & Value Bet AI (Version 3)

A professional-grade automated betting analysis system that identifies high-value opportunities across Football, Basketball, and Hockey. Integrated with Telegram for real-time analysis via manual match input or SportyBet booking codes.

## 🌟 Key Features

- **Multi-Sport Support:** Optimized pipelines for Football (Premier League, etc.), Basketball (NBA), and Hockey (NHL).
- **Playwright Booking Reader:** Automatically extracts matches and odds from SportyBet Nigeria using a simple booking code.
- **Automated Data Pipeline:**
  - **API-Sports Integration:** Fetches real-time fixtures, injuries, and player stats.
  - **Hybrid Scraping:** Multi-source automated scraping (Understat, FBref, SofaScore, etc.) for advanced metrics like xG, PPDA, and player efficiency.
  - **Market Analysis:** Integrated with OddsAPI to compute implied probabilities and identify value.
- **XGBoost Predictive Engine:** Uses a machine learning framework to calculate the probability of outcomes and identifies "Value Bets" (Model Confidence - Market Odds > 8%).
- **Clean Telegram Interface:**
  - Manual trigger via `/predict` command.
  - Real-time analysis of pasted matches or booking codes.
- **Robustness & Security:**
  - Async Playwright for non-blocking betslip extraction.
  - Strict Telegram authorization (responses restricted to authorized `CHAT_ID`).
  - Modular, scalable codebase for easy feature expansion.

## 📋 Prerequisites

- Python 3.8+
- [API-Sports](https://dashboard.api-football.com/) Key
- [The-Odds-API](https://the-odds-api.com/) Key
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd football-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright Browsers:**
   ```bash
   playwright install chromium --with-deps
   ```

4. **Configure Environment:**
   Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and Telegram credentials
   ```

## 🚀 Usage

Run the main orchestrator:
```bash
python main.py
```

### Telegram Commands & Formats

| Action | Format |
| :--- | :--- |
| **Start Bot** | `/start` |
| **Manual Predict** | `/predict <sport> <league_id> <season>` |
| **Booking Code Analysis** | `CODE | BC123XYZ` |
| **Manual Match Analysis** | `Sport | TeamA vs TeamB | OddsH | OddsD | OddsA` |

*Note: For non-football sports, the Draw odds (`OddsD`) are optional.*

## 🏗️ System Architecture

1. **`full_data_pipeline.py`**: Handles API interactions and multi-source web scraping.
2. **`extractor.py`**: Pure Async Playwright automation for SportyBet betslip extraction.
3. **`v3_prediction_model.py`**: Core ML logic, feature scaling, and value detection.
4. **`main.py`**: Async Telegram bot handlers.

## 🧪 Testing

The system includes a comprehensive test suite:
```bash
# Run unit and integration tests
python3 test_predictor.py
python3 verify_main.py
python3 test_manual_input.py
python3 test_playwright_integration.py
```

## 🔒 Security Note

This bot is configured to only respond to the `CHAT_ID` specified in your `.env` file. Any messages from unauthorized users will be silently ignored. Always keep your `.env` file private and never commit it to version control.
