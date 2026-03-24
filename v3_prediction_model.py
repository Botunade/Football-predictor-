import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Placeholder for a trained model and scaler
class MockModel:
    def predict_proba(self, X):
        # Return random probabilities for 3 classes: Home, Draw, Away
        batch_size = X.shape[0]
        probs = np.random.dirichlet(np.ones(3), size=batch_size)
        return probs

class MockScaler:
    def transform(self, X):
        return X

# Load actual model/scaler if they exist, otherwise use mocks
if os.path.exists("model_v3.pkl"):
    with open("model_v3.pkl", "rb") as f:
        model = pickle.load(f)
else:
    model = MockModel()

if os.path.exists("scaler_v3.pkl"):
    with open("scaler_v3.pkl", "rb") as f:
        scaler = pickle.load(f)
else:
    scaler = MockScaler()

def train_model(X_train, y_train):
    """
    Train an XGBoost model on historical data.
    X_train: DataFrame or array of features
    y_train: target (0: home, 1: draw, 2: away)
    """
    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
    clf.fit(X_train, y_train)

    # Save model
    with open("model_v3.pkl", "wb") as f:
        pickle.dump(clf, f)

    return clf

def predict_matches(matches_features):
    """Predict multiple matches."""
    return [predict_match(f) for f in matches_features]

def predict_match(features):
    """
    Predict match outcome (H2H, BTTS, Over 2.5) and calculate value.
    """
    # Define expected feature order for H2H
    feature_names = [
        "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
        "home_ppda", "away_ppda", "home_injury", "away_injury",
        "home_fatigue", "away_fatigue", "home_away_advantage",
        "rest_days_diff", "is_derby"
    ]

    X = np.array([[features.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)

    # H2H Probabilities
    probs = model.predict_proba(X_scaled)[0]
    model_prob_home = probs[0]

    # Value for H2H Home
    odds_home = features.get("odds_home")
    implied_prob_home = 1 / odds_home if odds_home and odds_home > 0 else 0
    value_home = model_prob_home - implied_prob_home

    # BTTS Prediction (Simplified Mock Logic)
    # In a real scenario, this would use a separate trained model
    btts_prob = (features.get("home_xG", 1.5) + features.get("away_xG", 1.5)) / 4.0
    btts_prob = min(max(btts_prob, 0.1), 0.9)

    odds_btts = features.get("odds_btts")
    implied_prob_btts = 1 / odds_btts if odds_btts and odds_btts > 0 else 0
    value_btts = btts_prob - implied_prob_btts

    # Over 2.5 Prediction (Simplified Mock Logic)
    over_25_prob = (features.get("home_xG", 1.5) + features.get("away_xG", 1.5)) / 3.5
    over_25_prob = min(max(over_25_prob, 0.1), 0.9)

    odds_over_25 = features.get("odds_over_25")
    implied_prob_over_25 = 1 / odds_over_25 if odds_over_25 and odds_over_25 > 0 else 0
    value_over_25 = over_25_prob - implied_prob_over_25

    return {
        "h2h": {
            "model_probability": round(model_prob_home, 3),
            "implied_probability": round(implied_prob_home, 3),
            "value": round(value_home, 3)
        },
        "btts": {
            "model_probability": round(btts_prob, 3),
            "implied_probability": round(implied_prob_btts, 3),
            "value": round(value_btts, 3)
        },
        "over_25": {
            "model_probability": round(over_25_prob, 3),
            "implied_probability": round(implied_prob_over_25, 3),
            "value": round(value_over_25, 3)
        }
    }
