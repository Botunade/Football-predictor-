import os
from dotenv import load_dotenv
from pathlib import Path

# Reference placeholders (for warnings only)
REF_TOKEN = os.getenv("REF_TELEGRAM_TOKEN", "your_token_here")
REF_CHAT_ID = os.getenv("REF_CHAT_ID", "your_chat_id_here")

# .env path
ENV_PATH = Path("./.env")

def verify_creds():
    """Checks TELEGRAM_TOKEN and CHAT_ID in .env without overwriting."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH)
        current_token = os.getenv("TELEGRAM_TOKEN")
        current_chat = os.getenv("CHAT_ID")
    else:
        current_token = None
        current_chat = None

    # Check token
    if not current_token or current_token == REF_TOKEN:
        print(f"[!] TELEGRAM_TOKEN is missing or placeholder. Please set your real token in .env.")
    else:
        print(f"[✓] TELEGRAM_TOKEN is set correctly.")

    # Check chat ID
    if not current_chat or current_chat == REF_CHAT_ID:
        print(f"[!] CHAT_ID is missing or placeholder. Please set your real chat ID in .env.")
    else:
        print(f"[✓] CHAT_ID is set correctly.")

    print(f"Current TELEGRAM_TOKEN: {current_token}")
    print(f"Current CHAT_ID: {current_chat}")

if __name__ == "__main__":
    verify_creds()
