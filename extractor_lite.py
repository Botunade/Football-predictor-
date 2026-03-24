import os
import time
import random
import re
from typing import List, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from dotenv import load_dotenv

load_dotenv()

CHROMIUM_PATH = os.getenv("CHROMIUM_PATH", "/data/data/com.termux/files/usr/bin/chromium")
FIREFOX_PATH = os.getenv("FIREFOX_PATH", "/data/data/com.termux/files/usr/bin/firefox")
GECKODRIVER_PATH = os.getenv("GECKODRIVER_PATH", "/data/data/com.termux/files/usr/bin/geckodriver")

def extract_booking_code_data_lite(code: str, retries: int = 2) -> List[Dict]:
    """
    Termux-optimized extractor using Selenium and native Python structures.
    """
    for attempt in range(retries + 1):
        try:
            return _execute_selenium_lite(code)
        except Exception as e:
            print(f"Lite extraction attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(5)
    return []

def _execute_selenium_lite(code: str) -> List[Dict]:
    results = []
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service

    ff_options = Options()
    ff_options.add_argument("--headless")

    driver_kwargs = {"options": ff_options}
    if os.path.exists(GECKODRIVER_PATH):
        driver_kwargs["service"] = Service(GECKODRIVER_PATH)

    if os.path.exists(FIREFOX_PATH):
        ff_options.binary_location = FIREFOX_PATH

    driver = webdriver.Firefox(**driver_kwargs)
    try:
        # SportyBet Nigeria
        driver.get("https://www.sportybet.com/ng")
        time.sleep(5)

        # Find and fill booking code
        try:
            input_box = driver.find_element(By.CSS_SELECTOR, 'input[placeholder="Booking Code"]')
            input_box.send_keys(code)

            load_button = driver.find_element(By.XPATH, '//button[contains(text(),"Load")]')
            load_button.click()
            time.sleep(7)

            # Extract matches
            elements = driver.find_elements(By.CSS_SELECTOR, ".m-bet-item, .match-row, .betslip-item")
            for el in elements:
                parsed = _parse_text_lite(el.text)
                if parsed: results.append(parsed)
        except Exception as e:
            print(f"Selenium parse error: {e}")

    finally:
        driver.quit()
    return results

def _parse_text_lite(text: str) -> Dict:
    if not text or "vs" not in text.lower():
        return None

    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Simple team extraction
    teams_line = next((l for l in lines if ' vs ' in l), None)
    if not teams_line: return None

    teams = teams_line.split(' vs ')
    home = teams[0].strip()
    away = teams[1].split(' ')[0].strip()

    # Simple odds extraction
    odds = []
    for line in lines:
        try:
            val = float(line.replace('NGN', '').strip())
            if 1.0 <= val <= 100.0: odds.append(val)
        except ValueError:
            continue

    return {
        "home": home,
        "away": away,
        "odds_home": odds[0] if len(odds) > 0 else 0.0,
        "odds_draw": odds[1] if len(odds) >= 3 else None,
        "odds_away": odds[2] if len(odds) >= 3 else (odds[1] if len(odds) == 2 else 0.0),
    }
