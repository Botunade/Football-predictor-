"""
Multi-Sport Predictor V4 - Environment & AI Credential Verification
Checks environment configuration and AI API key availability.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path("./.env")


def verify_creds():
    """Checks for AI API keys in .env without leaking secrets."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        print(f"[✓] Loaded configuration from {ENV_PATH}")
    else:
        print(f"[!] No .env file found at {ENV_PATH}. Using system environment variables.")

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("AI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    has_any_ai = False
    if gemini_key:
        masked = gemini_key[:4] + "..." + gemini_key[-4:] if len(gemini_key) > 8 else "***"
        print(f"[✓] Gemini API Key detected: {masked}")
        has_any_ai = True
    else:
        print("[i] GEMINI_API_KEY: Not configured (add GEMINI_API_KEY=your_key in .env to activate Gemini AI Agent)")

    if openai_key:
        masked = openai_key[:4] + "..." + openai_key[-4:] if len(openai_key) > 8 else "***"
        print(f"[✓] OpenAI API Key detected: {masked}")
        has_any_ai = True

    if anthropic_key:
        masked = anthropic_key[:4] + "..." + anthropic_key[-4:] if len(anthropic_key) > 8 else "***"
        print(f"[✓] Anthropic API Key detected: {masked}")
        has_any_ai = True

    if not has_any_ai:
        print("\n[NOTE] No AI API Key found. System will use statistical & heuristic fallback until key is added.")
    else:
        print("\n[✓] AI Analytical Agent is ready for live inference!")


if __name__ == "__main__":
    verify_creds()
