import os
from dotenv import load_dotenv, set_key
from pathlib import Path

# Reference values
REF_TOKEN = "8799260811:AAFNkyF1j-jooGhYu-IQX8xZmirTnAP5IvQ"
REF_CHAT_ID = "7104386905"

# .env path
ENV_PATH = Path("./.env")

def verify_and_correct_creds():
    """Checks and corrects TELEGRAM_TOKEN and CHAT_ID in the .env file."""
    # Load existing .env if present
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
        current_token = os.getenv("TELEGRAM_TOKEN")
        current_chat = os.getenv("CHAT_ID")
    else:
        current_token = None
        current_chat = None

    needs_reload = False

    # Check token
    if current_token != REF_TOKEN:
        print(f"[!] TELEGRAM_TOKEN mismatch or missing. Updating to reference value.")
        ENV_PATH.touch(exist_ok=True)  # create if not exist
        set_key(str(ENV_PATH), "TELEGRAM_TOKEN", REF_TOKEN)
        needs_reload = True
    else:
        print(f"[✓] TELEGRAM_TOKEN is correct.")

    # Check chat ID
    if current_chat != REF_CHAT_ID:
        print(f"[!] CHAT_ID mismatch or missing. Updating to reference value.")
        ENV_PATH.touch(exist_ok=True)
        set_key(str(ENV_PATH), "CHAT_ID", REF_CHAT_ID)
        needs_reload = True
    else:
        print(f"[✓] CHAT_ID is correct.")

    if needs_reload:
        load_dotenv(dotenv_path=ENV_PATH, override=True)

    print(f"Final TELEGRAM_TOKEN: {os.getenv('TELEGRAM_TOKEN')}")
    print(f"Final CHAT_ID: {os.getenv('CHAT_ID')}")

if __name__ == "__main__":
    verify_and_correct_creds()
