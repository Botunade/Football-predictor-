import os
import asyncio
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

from extractor_lite import extract_booking_code_data_lite
from pipeline_lite import build_features_lite
from model_lite import predict_match_lite
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path=".env")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

print("DEBUG TOKEN:", TELEGRAM_TOKEN)
# Explicitly load from .env for Termux compatibility
load_dotenv(dotenv_path=".env", override=True)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Token Validation
if not TELEGRAM_TOKEN or len(TELEGRAM_TOKEN) < 10:
    print(f"❌ ERROR: Invalid or missing TELEGRAM_TOKEN in .env file.")
    print(f"Current Value: '{TELEGRAM_TOKEN}'")
    exit(1)

print(f"DEBUG TOKEN: {TELEGRAM_TOKEN[:5]}...{TELEGRAM_TOKEN[-5:]}")

def format_lite_message(results):
    if not results: return "⚠️ No high-value bets found (Lite)."
    msg = "📊 *High-Value Bets* (V3 Lite)\n\n"
    for i, r in enumerate(results, 1):
        market = r['markets']['h2h']
        msg += f"{i}. *{r['match']}*\n"
        msg += f"   Outcome: {market['outcome']} | Value: {market['value']*100:.1f}%\n"
        msg += f"   Code: `{market['betting_code']}`\n\n"
    return msg

async def handle_lite_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != CHAT_ID: return
    text = update.message.text.strip()

    if text.startswith("CODE |"):
        code = text.split("|")[1].strip()
        await update.message.reply_text(f"⏳ (Lite) Extracting `{code}`...")
        matches = extract_booking_code_data_lite(code)

        if not matches:
            await update.message.reply_text("❌ No matches found or invalid code.")
            return

        results = []
        for m in matches:
            feats = build_features_lite(m)
            pred = predict_match_lite(feats)
            if pred['h2h']['value'] > 0.08:
                results.append({"match": f"{m['home']} vs {m['away']}", "markets": pred})

        await update.message.reply_text(format_lite_message(results), parse_mode='Markdown')
        return

    # Handle Manual: Sport | Match | Odds
    try:
        parts = [p.strip() for p in text.split('|')]
        if len(parts) >= 3:
            sport = parts[0].lower()
            match_str = parts[1]
            odds_h = float(parts[2])
            odds_d = float(parts[3]) if len(parts) > 3 else None
            odds_a = float(parts[4]) if len(parts) > 4 else 0.0

            await update.message.reply_text(f"⏳ Analyzing `{match_str}` (Lite)...")

            teams = match_str.split(' vs ')
            pseudo_m = {
                "home": teams[0].strip(),
                "away": teams[1].strip(),
                "odds_home": odds_h,
                "odds_draw": odds_d,
                "odds_away": odds_a
            }
            feats = build_features_lite(pseudo_m, sport)
            pred = predict_match_lite(feats, sport)

            await update.message.reply_text(format_lite_message([{"match": match_str, "markets": pred}]), parse_mode='Markdown')
    except Exception as e:
        pass

async def start_lite(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🚀 Predictor V3 Lite active! (No pandas/XGBoost)\nUse `CODE | <booking_code>`")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_lite))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_lite_input))
    print("Lite Bot running on Termux...")
    app.run_polling()
