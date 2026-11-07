"""
Data Validation Utilities
Ensures data quality and consistency across the prediction pipeline
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __bool__(self):
        return self.is_valid


def validate_match_data(match: Dict[str, Any]) -> ValidationResult:
    """
    Validate match data dictionary for required fields and data types

    Args:
        match: Dictionary containing match data

    Returns:
        ValidationResult with is_valid flag and any errors/warnings
    """
    errors = []
    warnings = []

    # Required fields
    required_fields = ["home_team", "away_team"]
    for field in required_fields:
        if field not in match:
            errors.append(f"Missing required field: {field}")
        elif not match[field]:
            errors.append(f"Empty value for required field: {field}")

    # Validate team names
    for team_field in ["home_team", "away_team"]:
        if team_field in match:
            team = match[team_field]
            if "Unknown" in str(team):
                errors.append(f"Placeholder team name in {team_field}: {team}")
            if len(str(team)) > 100:
                warnings.append(f"Very long team name in {team_field}")

    # Validate odds
    for odds_field in ["odds_home", "odds_draw", "odds_away"]:
        if odds_field in match:
            odds = match[odds_field]
            if odds is not None:
                try:
                    odds = float(odds)
                    if odds < 0:
                        errors.append(f"Negative odds in {odds_field}: {odds}")
                    if odds == 0:
                        warnings.append(f"Zero odds in {odds_field} (may indicate missing data)")
                except (ValueError, TypeError):
                    errors.append(f"Invalid odds value in {odds_field}: {match[odds_field]}")

    # Validate start_time format if present
    if "start_time" in match and match["start_time"]:
        start_time = match["start_time"]
        if not isinstance(start_time, str):
            errors.append(f"start_time must be string, got {type(start_time)}")
        else:
            # Try to parse ISO format
            try:
                # Handle various ISO formats
                patterns = [
                    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'
                ]
                is_valid_time = any(re.search(p, start_time) for p in patterns)
                if not is_valid_time:
                    warnings.append(f"Non-standard datetime format: {start_time}")
            except Exception:
                warnings.append(f"Could not validate datetime: {start_time}")

    is_valid = len(errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def validate_features(features: Dict[str, Any]) -> ValidationResult:
    """
    Validate ML feature dictionary

    Args:
        features: Dictionary containing ML features

    Returns:
        ValidationResult with is_valid flag and any errors/warnings
    """
    errors = []
    warnings = []

    # Check for NaN or infinite values
    for key, value in features.items():
        if isinstance(value, (int, float)):
            if value != value:  # NaN check
                errors.append(f"NaN value in feature: {key}")
            if value == float('inf') or value == float('-inf'):
                errors.append(f"Infinite value in feature: {key}")

    # Check for missing critical features (non-fatal warnings)
    critical_features = ["home_team", "away_team"]
    for feat in critical_features:
        if feat not in features or not features.get(feat):
            warnings.append(f"Missing critical feature: {feat}")

    is_valid = len(errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def normalize_odds(odds: float, min_odds: float = 1.01, max_odds: float = 100.0) -> float:
    """
    Normalize odds to reasonable range

    Args:
        odds: Raw odds value
        min_odds: Minimum valid odds
        max_odds: Maximum valid odds

    Returns:
        Normalized odds or 0 if invalid
    """
    if odds is None:
        return 0.0

    try:
        odds = float(odds)
        if odds < min_odds:
            return 0.0  # Invalid odds
        if odds > max_odds:
            return max_odds  # Cap extremely high odds
        return odds
    except (ValueError, TypeError):
        return 0.0


def clean_team_name(name: str) -> str:
    """
    Clean and standardize team name

    Args:
        name: Raw team name

    Returns:
        Cleaned team name
    """
    if not name:
        return "Unknown"

    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name.strip())

    # Remove common prefixes/suffixes that might be scraped incorrectly
    prefixes = ['[', ']', '•', '—', '-', '(', ')', '{', '}']
    for p in prefixes:
        name = name.replace(p, '')

    return name.strip()


def validate_prediction_result(prediction: Dict[str, Any]) -> ValidationResult:
    """
    Validate that a prediction result has all required fields

    Args:
        prediction: Prediction dictionary

    Returns:
        ValidationResult
    """
    errors = []
    warnings = []

    required_markets = ["h2h"]

    for market in required_markets:
        if market not in prediction:
            errors.append(f"Missing prediction market: {market}")
            continue

        market_data = prediction[market]
        required_fields = ["match", "outcome", "value", "model_probability", "implied_probability"]

        for field in required_fields:
            if field not in market_data:
                errors.append(f"Missing field {field} in {market} prediction")
            elif not isinstance(market_data, dict):
                errors.append(f"{market} prediction must be a dictionary")

    is_valid = len(errors) == 0

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


def check_feature_completeness(features: Dict[str, Any], required_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if all required features are present

    Args:
        features: Feature dictionary
        required_features: List of required feature names

    Returns:
        Tuple of (is_complete, missing_features)
    """
    missing = []
    for feat in required_features:
        if feat not in features or features[feat] is None:
            missing.append(feat)

    return len(missing) == 0, missing