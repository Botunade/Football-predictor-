from typing import List, Dict

def predict_match_lite(features: Dict, sport: str = "football") -> Dict:
    """
    Heuristic-based analysis for Lite version (No XGBoost).
    """
    home_xg = features.get("home_xG", 1.5)
    away_xg = features.get("away_xG", 1.5)

    # Very simple heuristic: model prob based on xG ratio
    total_xg = home_xg + away_xg
    model_prob_home = home_xg / total_xg if total_xg > 0 else 0.4

    odds_home = features.get("odds_home", 0)
    implied_prob_home = 1 / odds_home if odds_home > 0 else 0
    value_home = model_prob_home - implied_prob_home

    match_str = f"{features['home']} vs {features['away']}"

    return {
        "h2h": {
            "match": match_str,
            "outcome": "Home",
            "value": round(value_home, 3),
            "model_probability": round(model_prob_home, 3),
            "implied_probability": round(implied_prob_home, 3),
            "betting_code": match_str.replace(' ', '_').upper()
        }
    }

def analyze_matches_lite(matches_features: List[Dict], sport: str = "football") -> List[Dict]:
    results = []
    for f in matches_features:
        pred = predict_match_lite(f, sport)
        # Check for value > 8%
        if pred["h2h"]["value"] > 0.08:
            results.append({
                "match": f"{f['home']} vs {f['away']}",
                "markets": pred
            })
    return results
