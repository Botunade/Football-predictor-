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

def predict_match(features):
    """
    Predict match outcome and calculate value.
    features: dict of match features
    """
    # Define expected feature order
    feature_names = [
        "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
        "home_ppda", "away_ppda", "home_injury", "away_injury",
        "home_fatigue", "away_fatigue", "home_away_advantage",
        "rest_days_diff", "is_derby"
    ]

    X = np.array([[features.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[0]

    # Home Win = Index 0
    model_prob = probs[0]

    # Calculate Implied Probability from odds
    odds_home = features.get("odds_home")
    if odds_home and odds_home > 0:
        implied_prob = 1 / odds_home
        value = model_prob - implied_prob
    else:
        implied_prob = 0
        value = 0

    return {
        "model_probability": round(model_prob, 3),
        "implied_probability": round(implied_prob, 3),
        "value": round(value, 3)
    }
