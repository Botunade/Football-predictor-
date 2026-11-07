"""
Feature Scaling Module
Handles normalization and standardization of features for ML models
"""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    StandardScaler-like scaler with persistence support
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "FeatureScaler":
        """
        Compute mean and std for scaling

        Args:
            X: Training data (n_samples, n_features)

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply standardization

        Args:
            X: Data to transform

        Returns:
            Transformed data
        """
        if not self._fitted:
            logger.warning("Scaler not fitted, returning data unchanged")
            return np.asarray(X, dtype=float)

        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            X: Training data

        Returns:
            Transformed data
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform (denormalize)

        Args:
            X: Normalized data

        Returns:
            Original scale data
        """
        if not self._fitted:
            return np.asarray(X, dtype=float)

        X = np.asarray(X, dtype=float)
        return X * self.std_ + self.mean_

    def get_params(self) -> dict:
        """Get scaler parameters for serialization"""
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "_fitted": self._fitted
        }

    def set_params(self, params: dict) -> None:
        """Set scaler parameters from serialized state"""
        if params.get("mean"):
            self.mean_ = np.array(params["mean"])
        if params.get("std"):
            self.std_ = np.array(params["std"])
        self._fitted = params.get("_fitted", False)


class RobustScaler:
    """
    Alternative scaler using median and IQR
    More robust to outliers
    """

    def __init__(self):
        self.median_: Optional[np.ndarray] = None
        self.q1_: Optional[np.ndarray] = None
        self.q3_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "RobustScaler":
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        self.q1_ = np.percentile(X, 25, axis=0)
        self.q3_ = np.percentile(X, 75, axis=0)
        # Avoid division by zero
        iqr = self.q3_ - self.q1_
        iqr[iqr == 0] = 1.0
        self._iqr = iqr
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.asarray(X, dtype=float)

        X = np.asarray(X, dtype=float)
        return (X - self.median_) / self._iqr


class MinMaxScaler:
    """
    Min-Max scaler for normalizing features to [0, 1] range
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=float)
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        # Handle constant features
        constant_mask = (self.min_ == self.max_)
        self.min_[constant_mask] = 0.0
        self.max_[constant_mask] = 1.0
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.asarray(X, dtype=float)

        X = np.asarray(X, dtype=float)
        data_min, data_max = self.min_, self.max_
        feature_range = self.feature_range

        # Scale to [0, 1]
        X_scaled = (X - data_min) / (data_max - data_min)
        # Scale to feature_range
        X_scaled = X_scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# Default scaler instance
default_scaler = FeatureScaler()