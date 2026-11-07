#!/usr/bin/env python3
"""
SportyBet Code Reader and Analyzer
Reads betting codes, extracts match data via Playwright, and analyzes for opportunities.
"""

import asyncio
import re
import json
from datetime import datetime, timezone
from typing import List, Dict, Optional
from playwright.async_api import async_playwright, Page
import os
import sys

# Configuration - Pure Scraping V4 Mode
# Telegram and OddsAPI completely removed


async def extract_sporty_code(code: str, retries: int = 3) -> tuple[List[Dict], str]:
    """
    Extract match data from SportyBet by booking code.
    Returns: (matches_list, page_url_or_error_message)
    """
    for attempt in range(retries):
        browser = None
        try:
            async with async_playwright() as p:
                # Launch browser with appropriate settings
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-setuid-sandbox']
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    locale='en-US'
                )
                
                # Add stealth script
                await context.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                    "Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});"
                    "Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});"
                )
                
                page = await context.new_page()
                
                # Navigate to SportyBet
                url = f"https://www.sportybet.com/ng/play/{code}"
                print(f"[Playwright] Navigating to: {url}")
                
                response = await page.goto(url, timeout=60000, wait_until="networkidle")
                await page.wait_for_timeout(3000)  # Wait for JS to render
                
                current_url = page.url
                print(f"[Playwright] Final URL: {current_url}")
                
                # Check for redirects (login required, invalid code, etc.)
                if "sportybet.com/ng/" == current_url or "login" in current_url:
                    return [], "Redirected to login - authentication required"
                
                # Check for error messages
                error_selectors = [
                    ".m-error-msg",
                    ".error-message",
                    ":has-text('Invalid code')",
                    ":has-text('expired')",
                    ":has-text('Invalid Booking Code')"
                ]
                
                for selector in error_selectors:
                    try:
                        error = await page.query_selector(selector)
                        if error:
                            text = (await error.inner_text()).strip()
                            return [], f"SportyBet error: {text}"
                    except:
                        pass
                
                # Extract matches from betslip
                matches = await scrape_matches_from_page(page, code)
                
                await browser.close()
                return matches, current_url
                
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            if browser:
                await browser.close()
            await asyncio.sleep(2)
    
    return [], "Failed to extract data after multiple attempts"


async def scrape_matches_from_page(page: Page, code: str) -> List[Dict]:
    """Scrape match data from the SportyBet betslip page"""
    matches = []
    
    try:
        # Multiple selectors for different SportyBet layouts
        item_selectors = [
            ".betslip-item",
            ".m-bet-item",
            "[data-testid='betslip-item']",
            ".bet-item",
            ".match-card"
        ]
        
        items = []
        for selector in item_selectors:
            items = await page.query_selector_all(selector)
            if items:
                print(f"[Scraper] Found {len(items)} matches using selector: {selector}")
                break
        
        if not items:
            print("[Warning] No match items found with current selectors")
            return matches
        
        for i, item in enumerate(items):
            try:
                match_data = await extract_match_details(item, code)
                if match_data:
                    matches.append(match_data)
            except Exception as e:
                print(f"[Error] Failed to parse match {i}: {e}")
                continue
        
    except Exception as e:
        print(f"[Scraper] Error during scraping: {e}")
    
    return matches


async def extract_match_details(item, code: str) -> Optional[Dict]:
    """Extract individual match details from a betslip item"""
    
    # Extract team names
    home_el = await item.query_selector(".home-team, .team-name:first-child, .m-team:first-child")
    away_el = await item.query_selector(".away-team, .team-name:last-child, .m-team:last-child")
    
    home_team = (await home_el.inner_text()).strip() if home_el else "Unknown"
    away_team = (await away_el.inner_text()).strip() if away_el else "Unknown"
    
    # Clean team names
    home_team = re.sub(r'\s+', ' ', home_team)
    away_team = re.sub(r'\s+', ' ', away_team)
    
    # Extract odds - try multiple patterns
    odds_home, odds_draw, odds_away = 0.0, 0.0, 0.0
    
    # Try structured odds selectors first
    odds_selectors = [
        (".odds-home", 0), (".home-odds", 0),
        (".odds-draw", 1), (".draw-odds", 1),
        (".odds-away", 2), (".away-odds", 2)
    ]
    
    odds_found = [0.0, 0.0, 0.0]
    for selector, idx in odds_selectors:
        el = await item.query_selector(selector)
        if el:
            try:
                text = (await el.inner_text()).strip()
                odds_found[idx] = float(text)
            except:
                pass
    
    odds_home, odds_draw, odds_away = odds_found
    
    # Fallback: try ordered odds numbers
    if odds_home == 0.0:
        odds_list = await item.query_selector_all(".odds-number, .m-odds, .odds")
        if len(odds_list) >= 3:
            try:
                odds_home = float((await odds_list[0].inner_text()).strip())
                odds_draw = float((await odds_list[1].inner_text()).strip())
                odds_away = float((await odds_list[2].inner_text()).strip())
            except:
                pass
    
    # Extract start time
    start_time_el = await item.query_selector("[data-start-time], .match-time, .start-time")
    start_time = datetime.now(timezone.utc).isoformat()
    if start_time_el:
        try:
            start_time = await start_time_el.get_attribute("data-start-time") or start_time
        except:
            pass
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "odds_home": odds_home,
        "odds_draw": odds_draw,
        "odds_away": odds_away,
        "start_time": start_time,
        "code": code,
        "extraction_date": datetime.now(timezone.utc).isoformat()
    }


async def analyze_match(match: Dict) -> Dict:
    """Perform basic analysis on a match"""
    home_odds = match.get('odds_home', 0)
    draw_odds = match.get('odds_draw', 0)
    away_odds = match.get('odds_away', 0)
    
    # Find best odds
    best_outcome = "Home"
    best_odds = home_odds
    
    if draw_odds > best_odds:
        best_outcome = "Draw"
        best_odds = draw_odds
    if away_odds > best_odds:
        best_outcome = "Away"
        best_odds = away_odds
    
    # Calculate implied probability
    implied_prob = 1 / best_odds if best_odds > 0 else 0
    
    # Simple value detection
    value_score = best_odds - (implied_prob * 4)  # Simple heuristic
    
    return {
        **match,
        "analysis": {
            "best_outcome": best_outcome,
            "best_odds": best_odds,
            "implied_probability": round(implied_prob, 4),
            "value_score": round(value_score, 3),
            "is_value_bet": value_score > 0.5,
            "recommendation": "Consider" if value_score > 0.5 else "Monitor"
        }
    }


async def process_betting_code(code: str) -> Dict:
    """Main processing function: extract and analyze a betting code"""
    print(f"\n{'='*60}")
    print(f"Processing SportyBet Code: {code}")
    print(f"{'='*60}\n")
    
    # Extract matches
    matches, url_or_error = await extract_sporty_code(code)
    
    if not matches:
        return {
            "status": "error",
            "message": url_or_error or "No matches found",
            "code": code
        }
    
    # Analyze each match
    analyzed_matches = []
    for match in matches:
        analysis = await analyze_match(match)
        analyzed_matches.append(analysis)
    
    return {
        "status": "success",
        "code": code,
        "url": url_or_error if "http" in str(url_or_error) else None,
        "matches_extracted": len(analyzed_matches),
        "matches": analyzed_matches
    }


def format_output(result: Dict) -> str:
    """Format the analysis result for display"""
    if result["status"] == "error":
        return f"❌ Error: {result['message']}"
    
    output = []
    output.append(f"✅ Successfully extracted {result['matches_extracted']} matches from code: {result['code']}")
    output.append("")
    
    for i, match in enumerate(result['matches'], 1):
        analysis = match.get('analysis', {})
        output.append(f"{i}. {match['home_team']} vs {match['away_team']}")
        output.append(f"   Odds: {match['odds_home']:.2f} | {match['odds_draw']:.2f} | {match['odds_away']:.2f}")
        output.append(f"   🎯 Best: {analysis.get('best_outcome', 'N/A')} @ {analysis.get('best_odds', 0):.2f}")
        output.append(f"   Value Score: {analysis.get('value_score', 0):.2f}")
        if analysis.get('is_value_bet'):
            output.append(f"   💰 RECOMMENDATION: {analysis.get('recommendation', 'Consider')} this bet!")
        output.append("")
    
    return "\n".join(output)


async def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python bet_analyzer.py <sporty_code>")
        print("Example: python bet_analyzer.py ABC123XYZ")
        print("\nOr run interactively:")
        code = input("EnterSportyBet code: ").strip()
        if not code:
            print("No code provided")
            return
    else:
        code = sys.argv[1]
    
    result = await process_betting_code(code)
    output = format_output(result)
    print(output)
    
    # Save results
    output_file = f"bet_analysis_{code}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())