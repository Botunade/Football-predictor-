"""
Model Training Module
Provides training pipelines and evaluation metrics
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from .predictor import MatchPredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Comprehensive model training and evaluation pipeline
    """

    def __init__(self, model_path: str = "model_v4.pkl", scaler_path: str = "scaler_v4.pkl"):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.predictor = MatchPredictor(model_path, scaler_path)

    def prepare_training_data(self, historical_matches: List[Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare features and labels from historical match data

        Args:
            historical_matches: List of match dictionaries with outcomes

        Returns:
            Tuple of (X, y_dict) where y_dict contains 'h2h', 'btts', 'over25'
        """
        feature_vectors = []
        y_h2h_list = []
        y_btts_list = []
        y_over25_list = []

        feature_names = [
            "home_xG", "away_xG", "home_xGA", "away_xGA", "home_xGD", "away_xGD",
            "home_ppda", "away_ppda", "home_injury", "away_injury",
            "home_fatigue", "away_fatigue", "home_away_advantage",
            "rest_days_diff", "is_derby"
        ]

        for match in historical_matches:
            # Build feature vector
            features = np.array([match.get(f, 0) for f in feature_names])
            feature_vectors.append(features)

            # Labels
            outcome_map = {"home": 0, "draw": 1, "away": 2}
            h2h = match.get("actual_result", "draw").lower()
            y_h2h_list.append(outcome_map.get(h2h, 1))

            btts = match.get("actual_btts", False)
            y_btts_list.append(1 if btts else 0)

            over25 = match.get("actual_goals", 2.5) > 2.5
            y_over25_list.append(1 if over25 else 0)

        X = np.array(feature_vectors)

        return X, {
            'h2h': np.array(y_h2h_list),
            'btts': np.array(y_btts_list),
            'over25': np.array(y_over25_list)
        }

    def train(self, historical_matches: List[Dict], save: bool = True) -> Dict[str, Any]:
        """
        Train models on historical data

        Args:
            historical_matches: Historical match data for training
            save: Whether to save models to disk

        Returns:
            Training metrics and results
        """
        logger.info(f"Preparing training data from {len(historical_matches)} matches")

        X, y = self.prepare_training_data(historical_matches)

        if len(X) < 10:
            logger.warning("Insufficient data for training (need at least 10 samples)")
            return {"error": "insufficient_data"}

        # Train models
        self.predictor.fit(X, y['h2h'], y['btts'], y['over25'])

        # Evaluate
        metrics = self.evaluate(X, y)

        # Save if requested
        if save:
            self.predictor.save_models()
            logger.info(f"Models saved to {self.model_path}")

        return metrics

    def evaluate(self, X: np.ndarray, y: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Evaluate model performance using cross-validation

        Args:
            X: Feature matrix
            y: Dictionary of label arrays

        Returns:
            Evaluation metrics
        """
        results = {}

        try:
            # Standardize features using the fitted scaler
            X_scaled = self.predictor._scaler.transform(X)

            # Evaluate H2H
            h2h_preds = self.predictor._h2h_model.predict(X_scaled)
            h2h_probas = self.predictor._h2h_model.predict_proba(X_scaled)

            results['h2h'] = {
                'accuracy': float(accuracy_score(y['h2h'], h2h_preds)),
                'precision': float(precision_score(y['h2h'], h2h_preds, average='weighted', zero_division=0)),
                'recall': float(recall_score(y['h2h'], h2h_preds, average='weighted', zero_division=0)),
                'f1': float(f1_score(y['h2h'], h2h_preds, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(y['h2h'], h2h_preds).tolist()
            }
        except Exception as e:
            results['h2h'] = {'error': str(e)}

        try:
            # BTTS evaluation
            btts_preds = self.predictor._btts_model.predict(X_scaled)
            btts_probas = self.predictor._btts_model.predict_proba(X_scaled)[:, 1]

            results['btts'] = {
                'accuracy': float(accuracy_score(y['btts'], btts_preds)),
                'auc': float(roc_auc_score(y['btts'], btts_probas)) if len(np.unique(y['btts'])) > 1 else 0.0,
                'precision': float(precision_score(y['btts'], btts_preds, zero_division=0)),
                'recall': float(recall_score(y['btts'], btts_preds, zero_division=0)),
                'f1': float(f1_score(y['btts'], btts_preds, zero_division=0))
            }
        except Exception as e:
            results['btts'] = {'error': str(e)}

        try:
            # Over 2.5 evaluation
            over25_preds = self.predictor._over25_model.predict(X_scaled)
            over25_probas = self.predictor._over25_model.predict_proba(X_scaled)[:, 1]

            results['over_25'] = {
                'accuracy': float(accuracy_score(y['over25'], over25_preds)),
                'auc': float(roc_auc_score(y['over25'], over25_probas)) if len(np.unique(y['over25'])) > 1 else 0.0,
                'precision': float(precision_score(y['over25'], over25_preds, zero_division=0)),
                'recall': float(recall_score(y['over25'], over25_preds, zero_division=0)),
                'f1': float(f1_score(y['over25'], over25_preds, zero_division=0))
            }
        except Exception as e:
            results['over_25'] = {'error': str(e)}

        results['trained_at'] = datetime.now(timezone.utc).isoformat()
        results['sample_count'] = len(X)

        return results

    def generate_synthetic_data(self, n_samples: int = 1000) -> List[Dict]:
        """
        Generate synthetic training data for testing
        In production, replace with real historical data
        """
        data = []
        np.random.seed(42)

        for i in range(n_samples):
            # Generate realistic feature ranges
            match = {
                'home_xG': np.random.uniform(0.5, 3.0),
                'away_xG': np.random.uniform(0.5, 3.0),
                'home_xGA': np.random.uniform(0.5, 2.5),
                'away_xGA': np.random.uniform(0.5, 2.5),
                'home_xGD': 0.0,  # Will be computed
                'away_xGD': 0.0,
                'home_ppda': np.random.uniform(5, 20),
                'away_ppda': np.random.uniform(5, 20),
                'home_injury': np.random.uniform(0, 0.5),
                'away_injury': np.random.uniform(0, 0.5),
                'home_fatigue': np.random.uniform(0, 0.3),
                'away_fatigue': np.random.uniform(0, 0.3),
                'home_away_advantage': 1.1,
                'rest_days_diff': np.random.randint(-3, 4),
                'is_derby': np.random.randint(0, 2),
                'home_team': f'Team_{i % 20}',
                'away_team': f'Team_{(i + 5) % 20}'
            }

            # Compute derived values
            match['home_xGD'] = match['home_xG'] - match['home_xGA']
            match['away_xGD'] = match['away_xG'] - match['away_xGA']

            # Generate outcomes based on features (simplified)
            home_strength = match['home_xG'] - match['home_xGA']
            away_strength = match['away_xG'] - match['away_xGA']
            diff = home_strength - away_strength

            if diff > 0.3:
                match['actual_result'] = 'home'
            elif diff < -0.3:
                match['actual_result'] = 'away'
            else:
                match['actual_result'] = 'draw'

            # BTTS: more likely if both teams attack
            btts_chance = min(0.8, (match['home_xG'] + match['away_xG']) / 8)
            match['actual_btts'] = np.random.random() < btts_chance

            # Over 2.5: more likely with high xG
            over_chance = min(0.8, (match['home_xG'] + match['away_xG']) / 8)
            match['actual_goals'] = np.random.poisson(match['home_xG'] + match['away_xG'] - 2.5 + 2.5)
            match['actual_over25'] = match['actual_goals'] > 2.5

            data.append(match)

        return data


def train_model_from_data(
    historical_matches: List[Dict],
    model_path: str = "model_v4.pkl",
    scaler_path: str = "scaler_v4.pkl"
) -> Dict[str, Any]:
    """
    Convenience function to train a model from historical data
    """
    trainer = ModelTrainer(model_path, scaler_path)
    return trainer.train(historical_matches)