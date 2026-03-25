import asyncio
import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from extractor import extract_sporty_code  # Playwright extraction
from verify_creds import verify_creds

# --- Stage 0: Verify Telegram Credentials ---
verify_creds()

# Load correct .env
load_dotenv(dotenv_path=".env", override=True)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# --- Helper Functions ---
def determine_status(start_time: str, end_time: str = None) -> str:
    """Determine if a game is upcoming, running, finished, or expired."""
    try:
        now = datetime.now(timezone.utc)
        start_dt = datetime.fromisoformat(start_time).replace(tzinfo=timezone.utc)

        if end_time:
            end_dt = datetime.fromisoformat(end_time).replace(tzinfo=timezone.utc)
        else:
            # Default game duration placeholder (2 hours)
            from datetime import timedelta
            end_dt = start_dt + timedelta(hours=2)

        if now < start_dt:
            return "upcoming"
        elif start_dt <= now <= end_dt:
            return "running"
        else:
            return "finished"
    except Exception as e:
        print(f"Error determining status: {e}")
        return "unknown"

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

def format_matches_with_status_and_odds(games: list) -> str:
    """Format the games for Telegram display."""
    messages = []
    for i, game in enumerate(games, 1):
        msg = (
            f"Match {i}: {game['home_team']} vs {game['away_team']} | "
            f"Status: {game['status']} | "
            f"Odds: {game.get('odds_home', 'N/A')} | "
            f"{game.get('odds_draw', 'N/A')} | "
            f"{game.get('odds_away', 'N/A')}"
        )
        messages.append(msg)
    return "\n".join(messages)

async def background_analysis(games: list):
    """Run ML predictions or API updates in the background."""
    for game in games:
        # Here you would integrate with v3_prediction_model logic
        print(f"[Background Analysis] Processing {game['home_team']} vs {game['away_team']}")
    await asyncio.sleep(0.1)  # prevent blocking

# --- Telegram Command Handlers ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != CHAT_ID: return
    await update.message.reply_text(
        "🚀 Multi-Sport Predictor V3 is active!\n\n"
        "Use `/sporty <code>` to analyze a SportyBet betslip.\n"
        "Example: `/sporty BC123XYZ`"
    )

async def handle_sporty_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle SportyBet booking code input."""
    if str(update.effective_chat.id) != CHAT_ID: return

    if not context.args:
        await update.message.reply_text("❌ Please provide a SportyBet code. Usage: `/sporty <code>`")
        return

    code = context.args[0]
    await update.message.reply_text(f"⏳ Extracting matches from SportyBet code: {code}...")

    games = await extract_sporty_code(code)

    if not games:
        await update.message.reply_text("❌ Invalid or expired booking code, or no matches found.")
        return

    # Determine status & include current odds
    for game in games:
        game["status"] = determine_status(game["start_time"], game.get("end_time"))
        # Odds are already in the game dict from extractor

    await update.message.reply_text(format_matches_with_status_and_odds(games))
    asyncio.create_task(background_analysis(games))

async def handle_manual_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Fallback for manual match data if needed.
    """
    if str(update.effective_chat.id) != CHAT_ID: return
    # Reuse extraction/analysis logic or prompt for correct format
    await update.message.reply_text("Please use `/sporty <code>` for automated analysis.")

# --- Main Application ---
def main():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN is missing.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("sporty", handle_sporty_code))

    print("Bot is running and listening for commands...")
    app.run_polling()

if __name__ == "__main__":
    main()
