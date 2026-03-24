import time
import random
import os
import re
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Switch Backends (Can also be controlled via .env)
USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "True").lower() == "true"
USE_SELENIUM = os.getenv("USE_SELENIUM", "False").lower() == "true"

# Termux path fallback
CHROMIUM_PATH = os.getenv("CHROMIUM_PATH", "/data/data/com.termux/files/usr/bin/chromium")

def extract_booking_code_data(code: str, retries: int = 2) -> List[Dict]:
    """
    Unified extractor supporting both Playwright and Selenium.
    Returns: List of structured match dicts.
    """
    for attempt in range(retries + 1):
        try:
            if USE_PLAYWRIGHT:
                return _extract_with_playwright(code)
            elif USE_SELENIUM:
                return _extract_with_selenium(code)
            else:
                print("Error: No backend selected.")
                return []
        except Exception as e:
            print(f"Extraction attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(random.uniform(5, 10))
    return []

def _extract_with_playwright(code: str) -> List[Dict]:
    from playwright.sync_api import sync_playwright
    results = []

    with sync_playwright() as p:
        launch_kwargs = {
            "headless": True,
            "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"]
        }
        if os.path.exists(CHROMIUM_PATH):
            launch_kwargs["executable_path"] = CHROMIUM_PATH

        browser = p.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            page.goto("https://www.sportybet.com/ng", timeout=60000)
            time.sleep(random.uniform(3, 5))

            input_selector = 'input[placeholder="Booking Code"]'
            page.wait_for_selector(input_selector, timeout=20000)
            page.fill(input_selector, code)

            page.get_by_role("button", name="Load").click()
            time.sleep(random.uniform(5, 7))

            match_elements = page.query_selector_all(".m-bet-item, .match-row, .betslip-item")
            for el in match_elements:
                parsed = _parse_raw_text(el.inner_text())
                if parsed: results.append(parsed)

        finally:
            browser.close()

    return results

def _extract_with_selenium(code: str) -> List[Dict]:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    results = []

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    driver_kwargs = {"options": chrome_options}
    if os.path.exists(CHROMIUM_PATH):
        driver_kwargs["service"] = Service(CHROMIUM_PATH)

    driver = webdriver.Chrome(**driver_kwargs)

    try:
        driver.get("https://www.sportybet.com/ng")
        time.sleep(5)

        input_box = driver.find_element(By.XPATH, '//input[@placeholder="Booking Code"]')
        input_box.send_keys(code)

        try:
            load_button = driver.find_element(By.XPATH, '//button[contains(text(),"Load")]')
        except:
            load_button = driver.find_element(By.CSS_SELECTOR, '.m-btn-load, button.load')

        load_button.click()
        time.sleep(7)

        match_elements = driver.find_elements(By.CSS_SELECTOR, ".m-bet-item, .match-row, .betslip-item")
        for el in match_elements:
            parsed = _parse_raw_text(el.text)
            if parsed: results.append(parsed)

    finally:
        driver.quit()

    return results

def _parse_raw_text(text: str) -> Dict:
    if not text or "vs" not in text.lower():
        return None

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    full_text = " ".join(lines)

    # Heuristic: Find teams line (containing 'vs')
    vs_lines = [l for l in lines if ' vs ' in l]
    home, away = None, None

    if vs_lines:
        teams_line = min(vs_lines, key=len)
        if ':' in teams_line:
            teams_line = teams_line.split(':')[-1].strip()

        teams = teams_line.split(' vs ')
        home = teams[0].strip()
        away = teams[1].split(' ')[0].strip() # Take first word
    else:
        match = re.search(r"(?:^|[:])\s*([^:]+?)\s+vs\s+([^\s]+)", full_text)
        if match:
            home, away = match.group(1).strip(), match.group(2).strip()
            if ':' in home: home = home.split(':')[-1].strip()

    if not home or not away:
        return None

    # Search for odds
    odds = []
    for line in lines:
        clean_line = line.replace('NGN', '').replace('$', '').strip()
        try:
            val = float(clean_line)
            if 1.0 <= val <= 100.0:
                odds.append(val)
        except ValueError:
            continue

    odds_h = odds[0] if len(odds) > 0 else 0.0
    odds_d = odds[1] if len(odds) >= 3 else None
    odds_a = odds[2] if len(odds) >= 3 else (odds[1] if len(odds) == 2 else 0.0)

    return {
        "sport": "football" if odds_d is not None else "basketball",
        "home": home,
        "away": away,
        "odds_home": odds_h,
        "odds_draw": odds_d,
        "odds_away": odds_a,
        "raw": text
    }
