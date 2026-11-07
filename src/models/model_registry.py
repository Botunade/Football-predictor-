"""
Model Registry Module
Manages multiple model versions and provides fallback logic
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List, Any

from .predictor import MatchPredictor

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing multiple model versions and configurations
    Supports versioning, rollback, and A/B testing
    """

    def __init__(self, registry_path: str = "models_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load existing registry or create new one"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        return {
            "versions": [],
            "current_version": "v1.0",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(self, version: str, model_path: str, scaler_path: str,
                       metrics: Dict[str, Any], tags: List[str] = None) -> str:
        """
        Register a new model version

        Args:
            version: Version string (e.g., "v2.0", "2024-01-15")
            model_path: Path to model file
            scaler_path: Path to scaler file
            metrics: Performance metrics
            tags: Optional tags for categorization

        Returns:
            Model ID
        """
        model_id = f"{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model_info = {
            "id": model_id,
            "version": version,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metrics": metrics,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "performance_score": self._compute_performance_score(metrics)
        }

        self.registry["versions"].append(model_info)
        self.registry["current_version"] = version
        self._save_registry()

        logger.info(f"Registered model version {version} as {model_id}")
        return model_id

    def get_model(self, version: Optional[str] = None) -> MatchPredictor:
        """
        Get a model by version, or current if not specified

        Args:
            version: Specific version to load

        Returns:
            MatchPredictor instance
        """
        if version:
            for v in self.registry["versions"]:
                if v["version"] == version:
                    return MatchPredictor(v["model_path"], v["scaler_path"])

        # Return current version
        for v in self.registry["versions"]:
            if v["version"] == self.registry["current_version"]:
                return MatchPredictor(v["model_path"], v["scaler_path"])

        # Create default model
        return MatchPredictor()

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        return self.registry["versions"]

    def get_best_model(self, metric: str = "h2h.f1") -> MatchPredictor:
        """Get the best performing model based on a metric"""
        if not self.registry["versions"]:
            return MatchPredictor()

        best = max(
            self.registry["versions"],
            key=lambda x: x.get("performance_score", 0)
        )
        return self.get_model(best["version"])

    def _compute_performance_score(self, metrics: Dict) -> float:
        """Compute a single performance score from metrics"""
        score = 0.0

        # Weight different metrics
        if "h2h" in metrics and isinstance(metrics["h2h"], dict):
            score += metrics["h2h"].get("f1", 0) * 0.4
            score += metrics["h2h"].get("accuracy", 0) * 0.3

        if "btts" in metrics and isinstance(metrics["btts"], dict):
            score += metrics["btts"].get("f1", 0) * 0.15

        if "over_25" in metrics and isinstance(metrics["over_25"], dict):
            score += metrics["over_25"].get("f1", 0) * 0.15

        return score

    def compare_models(self, versions: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        comparison = {}

        for v in self.registry["versions"]:
            if v["version"] in versions:
                comparison[v["version"]] = {
                    "performance_score": v["performance_score"],
                    "h2h_f1": v.get("metrics", {}).get("h2h", {}).get("f1", 0),
                    "btts_auc": v.get("metrics", {}).get("btts", {}).get("auc", 0),
                    "over25_f1": v.get("metrics", {}).get("over_25", {}).get("f1", 0),
                    "created_at": v["created_at"]
                }

        return comparison

    def rollback(self, version: str) -> bool:
        """Rollback to a previous version"""
        for v in self.registry["versions"]:
            if v["version"] == version:
                self.registry["current_version"] = version
                self._save_registry()
                logger.info(f"Rolled back to version {version}")
                return True
        return False


# Default registry instance
_default_registry = None


def get_registry() -> ModelRegistry:
    """Get or create default registry instance"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry


def get_model(version: Optional[str] = None) -> MatchPredictor:
    """Convenience function to get a model"""
    return get_registry().get_model(version)