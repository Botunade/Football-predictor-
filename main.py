import pandas as pd
import time
import os
import asyncio
import logging
from datetime import datetime
from telegram import Bot, Update
from telegram.error import TimedOut, NetworkError
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from verify_creds import verify_creds

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# === IMPORT YOUR MODULES / SCRIPTS ===
from full_data_pipeline import fetch_fixtures, fetch_odds, build_dataset, build_features
from v3_prediction_model import predict_match
from extractor import extract_booking_code_data

# === LOAD ENV VARS ===
load_dotenv()

# Verify credentials
verify_creds()

# Fetch from environment
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Check token and chat ID before starting
if TELEGRAM_TOKEN in (None, "", "your_token_here") or CHAT_ID in (None, "", "your_chat_id_here"):
    print("[!] Telegram token or chat ID is not properly set. Exiting.")
    exit(1)

# Initialize bot
try:
    bot = Bot(token=TELEGRAM_TOKEN)
except Exception as e:
    print(f"Warning: Could not initialize Telegram Bot. {e}")
    bot = None

# Helper function to send async message with retries
async def send_telegram_message(message, retries=3):
    if not bot:
        logger.warning(f"Bot not initialized. Message: {message}")
        return

    for attempt in range(retries):
        try:
            async with bot:
                await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')
            return
        except TimedOut:
            logger.warning(f"Timeout sending message (Attempt {attempt+1}/{retries}). Retrying...")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Telegram error sending message: {e}")
            break

async def safe_reply_text(update: Update, text: str, parse_mode='Markdown', retries=3):
    """Safely reply to a message with retry logic for timeouts."""
    for attempt in range(retries):
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
            return
        except TimedOut:
            logger.warning(f"Timeout replying to message (Attempt {attempt+1}/{retries}). Retrying...")
            await asyncio.sleep(2)
        except Exception as e:
            logger.error(f"Error in safe_reply_text: {e}")
            break

def format_telegram_message(results):
    """
    Formats prediction results for Telegram – cleanly listing only results and codes.
    """
    if not results:
        return "⚠️ No high-value bets found in this cycle."

    message = "📊 *High-Value Bets* (Automated V3)\n\n"

    for i, r in enumerate(results, start=1):
        for market_name, data in r['markets'].items():
            if data['value'] > 0.08:
                message += f"{i}. *{r['match']}* | Outcome: {data['outcome']} | Value: {data['value']*100:.1f}% | Code: `{data['betting_code']}`\n"

    message += "✅ Data source: APIs + Scraping + OddsAPI\n"
    message += "💡 Only high-value bets included (value > 8%)\n"

    return message

# === PIPELINE + MODEL FUNCTION ===
async def run_version3_pipeline(league_id, season, sport="football"):
    print(f"Starting pipeline run for {sport.capitalize()} League {league_id}, Season {season}")

    # 1. Fetch fixtures
    fixtures = fetch_fixtures(league_id, season, sport)
    if not fixtures:
        print("No fixtures found.")
        return

    # 2. Fetch odds
    odds_data = fetch_odds(sport, league_id)

    # 3. Build dataset (team + player + context metrics)
    dataset = build_dataset(fixtures, sport)

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
        prediction = predict_match(features, sport)

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
    await safe_reply_text(
        update,
        "🚀 *Multi-Sport Predictor V3* is active!\n\n"
        "Send match data in this format to analyze:\n"
        "`Sport | HomeTeam vs AwayTeam | OddsHome | OddsDraw | OddsAway | OptionalScrapeURL` (Draw is optional for non-football)\n"
        "Example: `Football | Arsenal vs Chelsea | 2.1 | 3.5 | 3.2`"
    )

async def handle_manual_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle multi-sport messages:
    1. CODE | <booking_code>
    2. Sport | Home vs Away | OddsH | OddsD | OddsA
    """
    if str(update.effective_chat.id) != CHAT_ID:
        return

    text = update.message.text.strip()

    # Check for Booking Code input
    if text.startswith("CODE |"):
        booking_code = text.split("|")[1].strip()
        await safe_reply_text(update, f"⏳ Extracting matches from SportyBet code: `{booking_code}`...")

        extracted_matches = await extract_booking_code_data(booking_code)

        if not extracted_matches:
            await safe_reply_text(update, "❌ Invalid or expired booking code, or no matches found.")
            return

        await safe_reply_text(update, f"✅ Extracted {len(extracted_matches)} matches. Analyzing...")

        results = []
        for match in extracted_matches:
            try:
                # 1. Map extracted data to our feature builder
                pseudo_fixture = {
                    "home_team": match["home"],
                    "away_team": match["away"],
                    "sport": match["sport"],
                    "fixture": {"id": 0},
                    "league": {"id": 0, "season": 2024},
                    "teams": {
                        "home": {"id": 0, "name": match["home"]},
                        "away": {"id": 0, "name": match["away"]}
                    }
                }

                # 2. Build features
                features = build_features(pseudo_fixture, match["sport"])

                # 3. Add extracted odds
                features.update({
                    "odds_home": match["odds_home"],
                    "odds_draw": match["odds_draw"],
                    "odds_away": match["odds_away"]
                })

                # 4. Predict
                prediction = predict_match(features, match["sport"])

                results.append({
                    "match": f"{match['home']} vs {match['away']}",
                    "markets": prediction
                })
            except Exception as e:
                print(f"Error analyzing extracted match: {e}")
                continue

        if results:
            message = format_telegram_message(results)
            await safe_reply_text(update, message)
        else:
            await safe_reply_text(update, "No high-value bets identified from this code.")
        return

    try:
        parts = [t.strip() for t in text.split("|")]
        if len(parts) < 3:
            raise ValueError("Invalid format. Use: Sport | Home vs Away | Odds...")

        sport = parts[0].lower()
        if sport not in ["football", "basketball", "hockey"]:
            raise ValueError(f"Sport '{sport}' not supported.")

        match_part = parts[1]
        teams = match_part.split(' vs ')
        if len(teams) != 2:
            raise ValueError("Match must be in 'HomeTeam vs AwayTeam' format.")

        home_team, away_team = teams[0].strip(), teams[1].strip()

        # Parsing odds
        odds_h = float(parts[2])
        odds_d = None
        odds_a = None

        if sport == "football":
            if len(parts) < 5:
                raise ValueError("Football requires 3 odds (H|D|A)")
            odds_d = float(parts[3])
            odds_a = float(parts[4])
        else:
            if len(parts) < 4:
                raise ValueError(f"{sport.capitalize()} requires 2 odds (H|A)")
            odds_a = float(parts[3])

        await safe_reply_text(update, f"⏳ *Automated Analysis:* {home_team} vs {away_team} ({sport.capitalize()})...")

        from full_data_pipeline import scrape_understat_team, scrape_advanced_stats

        # 1. Automatically fetch Scraped stats
        scraped_data = scrape_advanced_stats(sport, match_part)

        # 2. Base stats (Football-only Understat for now)
        if sport == "football":
            home_stats = scrape_understat_team(home_team, 2024)
            away_stats = scrape_understat_team(away_team, 2024)
        else:
            home_stats = {"xG": 0, "xGA": 0, "xGD": 0, "PPDA": 0}
            away_stats = {"xG": 0, "xGA": 0, "xGD": 0, "PPDA": 0}

        if scraped_data:
            home_stats.update(scraped_data.get("home", {}))
            away_stats.update(scraped_data.get("away", {}))

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
            "home_injury": 0.1,
            "away_injury": 0.1,
            "home_fatigue": 0.2,
            "away_fatigue": 0.2,
            "home_away_advantage": 1.1,
            "rest_days_diff": 0,
            "is_derby": 0,
            "odds_home": odds_h,
            "odds_draw": odds_d,
            "odds_away": odds_a
        }

        prediction = predict_match(features, sport)

        result = {
            "match": f"{home_team} vs {away_team}",
            "markets": prediction
        }

        message = format_telegram_message([result])
        await safe_reply_text(update, message)

    except Exception as e:
        await safe_reply_text(update, f"❌ Error: {str(e)}")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a Telegram message to notify the developer."""
    logger.error(f"Exception while handling an update: {context.error}")

    # Notify the authorized chat about the internal error
    try:
        error_msg = f"⚠️ *Internal Bot Error:*\n`{str(context.error)[:1000]}`"
        await bot.send_message(chat_id=CHAT_ID, text=error_msg, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Failed to send error notification: {e}")

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sport = context.args[0].lower() if len(context.args) > 0 else "football"
        league_id = int(context.args[1]) if len(context.args) > 1 else 39
        season = int(context.args[2]) if len(context.args) > 2 else 2024

        await safe_reply_text(update, f"⏳ Running {sport} pipeline for League {league_id}, Season {season}...")
        await run_version3_pipeline(league_id, season, sport)
    except (IndexError, ValueError):
        await safe_reply_text(update, "❌ Usage: /predict <sport> <league_id> <season>\nExample: /predict football 39 2024")

# === SCHEDULER / MAIN LOOP ===
async def scheduler_task(league_id, season, interval, sport="football"):
    while True:
        print(f"Running scheduled {sport} pipeline at {datetime.now()} ...")
        try:
            await run_version3_pipeline(league_id, season, sport)
        except Exception as e:
            print(f"Error during {sport} pipeline run:", e)
            await send_telegram_message(f"⚠️ {sport.capitalize()} Pipeline error: {e}")

        print(f"Sleeping {interval/3600} hours before next run...")
        await asyncio.sleep(interval)

def telegram_bot_handler():
    """Build the application handler."""
    if not TELEGRAM_TOKEN:
        print("Error: No TELEGRAM_TOKEN found in environment.")
        return None

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    return application

def main():
    # Build application with increased timeout
    app = ApplicationBuilder() \
        .token(TELEGRAM_TOKEN) \
        .read_timeout(30) \
        .connect_timeout(30) \
        .build()

    # Add handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_manual_input))

    # Add global error handler
    app.add_error_handler(error_handler)

    print("Bot is running and listening for commands...")
    # run_polling() internally handles the asyncio loop creation or reuse
    app.run_polling()

if __name__ == "__main__":
    main()
