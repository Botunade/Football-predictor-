import pandas as pd
import time
import os
import asyncio
from datetime import datetime
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# === IMPORT YOUR MODULES / SCRIPTS ===
from full_data_pipeline import fetch_fixtures, fetch_odds, build_dataset
from v3_prediction_model import predict_match

# === LOAD ENV VARS ===
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Initialize bot
try:
    bot = Bot(token=TELEGRAM_TOKEN)
except Exception as e:
    print(f"Warning: Could not initialize Telegram Bot. {e}")
    bot = None

# Helper function to send async message
async def send_telegram_message(message):
    if not bot:
        print(f"Bot not initialized. Message: {message}")
        return
    try:
        async with bot:
            await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
    except Exception as e:
        print(f"Telegram error: {e}")

def format_telegram_message(results):
    """
    Formats prediction results for Telegram.
    Input: results = list of dicts, each containing:
        - match
        - markets (dict of market results)
    Output: formatted string ready to send via bot.send_message
    """
    if not results:
        return "⚠️ No high-confidence matches found in this cycle."

    message = "📊 *High-Confidence Value Bets* (Version 3)\n\n"

    for i, r in enumerate(results, start=1):
        message += f"*{i}. Match:* {r['match']}\n"
        for market_name, data in r['markets'].items():
            if data['value'] > 0.08:
                message += f"• *Market:* {market_name.upper()}\n"
                message += f"  • Model Prob: {data['model_probability']:.2%} | Implied Prob: {data['implied_probability']:.2%}\n"
                message += f"  • Value: {data['value']:.2%}\n"

        message += f"• Betting Code: `{r['match'].replace(' ', '_').upper()}`\n\n"

    message += "✅ Data source: APIs + Scraping + OddsAPI\n"
    message += "💡 Only high-value bets included (value > 8%)\n"

    return message

# === PIPELINE + MODEL FUNCTION ===
async def run_version3_pipeline(league_id, season):
    print(f"Starting pipeline run for League {league_id}, Season {season}")

    # 1. Fetch fixtures
    fixtures = fetch_fixtures(league_id, season)
    if not fixtures:
        print("No fixtures found.")
        return

    # 2. Fetch odds
    odds_data = fetch_odds()

    # 3. Build dataset (team + player + context metrics)
    dataset = build_dataset(fixtures)

    # 4. Merge odds
    df_odds = pd.DataFrame(odds_data)
    if not df_odds.empty:
        final_dataset = dataset.merge(df_odds, on=["home_team", "away_team"], how="left")
    else:
        # If no odds data, ensure columns exist
        final_dataset = dataset
        final_dataset["odds_home"] = None
        final_dataset["odds_away"] = None
        final_dataset["odds_draw"] = None

    # 5. Run predictions & compute value
    results = []
    for _, match in final_dataset.iterrows():
        features = match.to_dict()
        prediction = predict_match(features)

        # Check if any market has high value
        has_value = any(m['value'] > 0.08 for m in prediction.values())

        if has_value:
            results.append({
                "match": f"{features['home_team']} vs {features['away_team']}",
                "markets": prediction
            })

    # 6. Send results to Telegram
    message = format_telegram_message(results)
    await send_telegram_message(message)

    print("Pipeline run complete.")

# === TELEGRAM COMMAND HANDLERS ===
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != CHAT_ID:
        return
    await update.message.reply_text(
        "🚀 *Football Predictor V3* is active!\n\n"
        "Send match data in this format to analyze:\n"
        "`HomeTeam vs AwayTeam | OddsHome | OddsDraw | OddsAway`",
        parse_mode='Markdown'
    )

async def handle_manual_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle messages in the format: HomeTeam vs AwayTeam | OddsHome | OddsDraw | OddsAway"""
    if str(update.effective_chat.id) != CHAT_ID:
        return

    text = update.message.text
    try:
        # Example input: Manchester United vs Liverpool | 2.50 | 3.40 | 2.80
        parts = [p.strip() for p in text.split('|')]
        if len(parts) != 4:
            raise ValueError("Invalid format. Use: HomeTeam vs AwayTeam | OddsHome | OddsDraw | OddsAway")

        teams = parts[0].split(' vs ')
        if len(teams) != 2:
            raise ValueError("Match must be in 'HomeTeam vs AwayTeam' format.")

        home_team, away_team = teams[0].strip(), teams[1].strip()
        odds_h = float(parts[1])
        odds_d = float(parts[2])
        odds_a = float(parts[3])

        await update.message.reply_text(f"⏳ Analyzing *{home_team} vs {away_team}*...", parse_mode='Markdown')

        # Try to parse optional extra markets from the text
        # If the input has more than 4 parts, we can look for BTTS/O2.5 odds
        # For simplicity, we assume the format could be: Match | H | D | A | BTTS_YES | O2.5
        odds_btts = None
        odds_o25 = None
        if len(parts) >= 5:
            odds_btts = float(parts[4])
        if len(parts) >= 6:
            odds_o25 = float(parts[5])

        # Create a single match result for prediction
        # In a real scenario, we'd fetch actual team data from our pipeline here
        # For now, we'll use our build_features/predict_matches modular flow
        from full_data_pipeline import scrape_understat_team, fetch_player_info, compute_context

        # Build features for this manual match
        home_stats = scrape_understat_team(home_team, 2024)
        away_stats = scrape_understat_team(away_team, 2024)

        features = {
            "home_team": home_team,
            "away_team": away_team,
            "home_xG": home_stats["xG"],
            "away_xG": away_stats["xG"],
            "home_xGA": home_stats["xGA"],
            "away_xGA": away_stats["xGA"],
            "home_xGD": home_stats["xGD"],
            "away_xGD": away_stats["xGD"],
            "home_ppda": home_stats["PPDA"],
            "away_ppda": away_stats["PPDA"],
            "home_injury": 0.1, # Default
            "away_injury": 0.1,
            "home_fatigue": 0.2,
            "away_fatigue": 0.2,
            "home_away_advantage": 1.1,
            "rest_days_diff": 0,
            "is_derby": 0,
            "odds_home": odds_h,
            "odds_draw": odds_d,
            "odds_away": odds_a,
            "odds_btts": odds_btts,
            "odds_over_25": odds_o25
        }

        prediction = predict_match(features)

        result = {
            "match": f"{home_team} vs {away_team}",
            "markets": prediction
        }

        message = format_telegram_message([result])
        await update.message.reply_text(message, parse_mode='Markdown')

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        league_id = int(context.args[0])
        season = int(context.args[1])
        await update.message.reply_text(f"⏳ Running pipeline for League {league_id}, Season {season}...")
        await run_version3_pipeline(league_id, season)
    except (IndexError, ValueError):
        await update.message.reply_text("❌ Usage: /predict <league_id> <season>\nExample: /predict 39 2024")

# === SCHEDULER / MAIN LOOP ===
async def scheduler_task(league_id, season, interval):
    while True:
        print(f"Running scheduled pipeline at {datetime.now()} ...")
        try:
            await run_version3_pipeline(league_id, season)
        except Exception as e:
            print("Error during pipeline run:", e)
            await send_telegram_message(f"⚠️ Pipeline error: {e}")

        print(f"Sleeping {interval/3600} hours before next run...")
        await asyncio.sleep(interval)

def telegram_bot_handler():
    """Build the application handler."""
    if not TELEGRAM_TOKEN:
        print("Error: No TELEGRAM_TOKEN found in environment.")
        return None

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    return application

async def main():
    # Build application
    application = telegram_bot_handler()
    if not application:
        return

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_manual_input))

    # Start the scheduler as a background task
    LEAGUE_ID = 39
    SEASON = 2024
    INTERVAL = 12 * 60 * 60

    asyncio.create_task(scheduler_task(LEAGUE_ID, SEASON, INTERVAL))

    print("Bot is running and listening for commands...")
    async with application:
        await application.initialize()
        await application.start()
        await application.updater.start_polling()

        # Keep the main loop alive
        while True:
            await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
