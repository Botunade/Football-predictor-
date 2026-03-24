import pandas as pd
import time
import os
import asyncio
from datetime import datetime
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
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
        - odds_home
        - odds_away
        - model_prob
        - implied_prob
        - value
    Output: formatted string ready to send via bot.send_message
    """
    if not results:
        return "⚠️ No high-confidence matches found in this cycle."

    message = "📊 *High-Confidence Value Bets* (Version 3)\n\n"

    for i, r in enumerate(results, start=1):
        message += f"*{i}. Match:* {r['match']}\n"
        message += f"• Odds H/A: {r['odds_home']} / {r['odds_away']}\n"
        message += f"• Model Prob: {r['model_prob']:.2%} | Implied Prob: {r['implied_prob']:.2%}\n"
        message += f"• Value: {r['value']:.2%}\n"
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

        # Threshold for high-confidence/value bet
        if prediction["value"] > 0.08:
            results.append({
                "match": f"{features['home_team']} vs {features['away_team']}",
                "odds_home": features.get("odds_home"),
                "odds_away": features.get("odds_away"),
                "model_prob": prediction["model_probability"],
                "implied_prob": prediction["implied_probability"],
                "value": prediction["value"]
            })

    # 6. Send results to Telegram
    message = format_telegram_message(results)
    await send_telegram_message(message)

    print("Pipeline run complete.")

# === TELEGRAM COMMAND HANDLERS ===
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Football Predictor V3 is active!\nUse /predict <league_id> <season> to trigger an immediate run.")

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

async def main():
    # Build application
    if not TELEGRAM_TOKEN:
        print("Error: No TELEGRAM_TOKEN found in environment.")
        return

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("predict", predict_command))

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
