"""JSON Export and Documentation System"""

import csv
import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings

from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# =============================================================================
# JSON Schema Definitions
# =============================================================================

class SchemaVersion(Enum):
    V1_0 = "1.0"
    V2_0 = "2.0"

@dataclass
class MatchSchema:
    home_team: str = ""
    away_team: str = ""
    league: str = ""
    date: str = ""
    time: str = ""
    venue: str = ""
    status: str = "upcoming"
    sport: str = "football"
    match_id: str = ""
    home_xG: float = 0.0
    away_xG: float = 0.0
    home_xGA: float = 0.0
    away_xGA: float = 0.0

    def to_dict(self):
        return asdict(self)

@dataclass
class OddsSchema:
    """Complete odds schema with all market types"""
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0
    over_2_5_odds: float = 0.0
    under_2_5_odds: float = 0.0
    over_1_5_odds: float = 0.0
    under_1_5_odds: float = 0.0
    btts_yes: float = 0.0
    btts_no: float = 0.0
    ah_home: float = 0.0
    ah_draw: float = 0.0
    ah_away: float = 0.0
    ah_home_line: float = 0.0
    ah_away_line: float = 0.0
    correct_score_0_0: float = 0.0
    correct_score_1_0: float = 0.0
    correct_score_0_1: float = 0.0
    correct_score_1_1: float = 0.0
    htft_home_home: float = 0.0
    htft_home_away: float = 0.0
    htft_away_away: float = 0.0
    source: str = "unknown"
    last_updated: str = ""

    def to_dict(self):
        return asdict(self)

    def get_h2h_odds(self):
        return {"home": self.home_odds, "draw": self.draw_odds, "away": self.away_odds}

    def get_market_odds(self, market):
        mapping = {
            "h2h": self.get_h2h_odds(),
            "ah": {"home": self.ah_home, "draw": self.ah_draw, "away": self.ah_away},
            "ou": {"over_2_5": self.over_2_5_odds, "under_2_5": self.under_2_5_odds},
            "btts": {"yes": self.btts_yes, "no": self.btts_no}
        }
        return mapping.get(market, {})

@dataclass
class PredictionResultSchema:
    """Prediction results schema"""
    match: str = ""
    outcome: str = "Home"
    value: float = 0.0
    betting_code: str = ""
    model_probability: float = 0.0
    implied_probability: float = 0.0
    is_value_bet: bool = False
    confidence_level: str = "low"

    def to_dict(self):
        return asdict(self)

@dataclass
class MetadataSchema:
    """Metadata schema including extraction and analysis timestamps"""
    run_id: str = ""
    version: str = "1.0"
    schema_version: str = "1.0"
    created_at: str = ""
    extraction_start: str = ""
    extraction_end: str = ""
    analysis_start: str = ""
    analysis_end: str = ""
    source: str = ""
    source_url: str = ""
    extraction_method: str = ""
    total_matches: int = 0
    total_predictions: int = 0
    processing_time_seconds: float = 0.0

    def to_dict(self):
        return asdict(self)

# =============================================================================
# Validation Functions
# =============================================================================

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def validate_match_schema(data: Dict[str, Any]) -> bool:
    """Validate match data against schema"""
    for field_name in ["home_team", "away_team"]:
        if field_name not in data or not data[field_name]:
            raise DataValidationError(f"Missing required field: {field_name}")
    for field_name in ["home_odds", "draw_odds", "away_odds"]:
        if field_name in data:
            odds = float(data[field_name])
            if odds < 0:
                raise DataValidationError(f"Negative odds in {field_name}")
    return True

def validate_odds_schema(data: Dict[str, Any]) -> bool:
    """Validate odds data against schema"""
    if not isinstance(data, dict):
        raise DataValidationError("Odds must be a dictionary")
    return True

def validate_prediction_schema(data: Dict[str, Any]) -> bool:
    """Validate prediction result against schema"""
    for field_name in ["match", "outcome", "value", "model_probability", "implied_probability"]:
        if field_name not in data:
            raise DataValidationError(f"Missing required field: {field_name}")
    return True

# =============================================================================
# JSON Exporter Class
# =============================================================================

class JSONExporter:
    """Main exporter class with schema validation, export formats, and documentation."""

    def __init__(self, output_dir: str = "./data", schema_version: str = "1.0"):
        self.output_dir = Path(output_dir)
        self.schema_version = schema_version
        self._run_id = hashlib.md5(
            f"{datetime.now().isoformat()}-{id(self)}".encode()
        ).hexdigest()[:12]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_matches_to_json(
        self, matches: List[Dict[str, Any]],
        predictions: Optional[List[Dict[str, Any]]] = None,
        odds_data: Optional[Dict[str, Any]] = None,
        source: str = "sportybet",
        output_file: Optional[str] = None,
        validate: bool = True
    ) -> Dict[str, Any]:
        """Export matches and predictions to JSON file."""
        if validate:
            for match in matches:
                validate_match_schema(match)
            if predictions:
                for pred in predictions:
                    validate_prediction_schema(pred)
            if odds_data:
                validate_odds_schema(odds_data)

        now = datetime.now(timezone.utc)
        extraction_end = now.isoformat()

        export_data = {
            "schema_version": self.schema_version,
            "metadata": {
                "run_id": self._run_id,
                "version": "1.0",
                "schema_version": self.schema_version,
                "created_at": now.isoformat(),
                "source": source,
                "total_matches": len(matches),
                "total_predictions": len(predictions) if predictions else 0
            },
            "extraction": {
                "start_time": now.isoformat(),
                "end_time": extraction_end,
                "method": "playwright"
            },
            "analysis": {
                "start_time": now.isoformat(),
                "end_time": extraction_end,
                "model_version": "v4.0"
            },
            "matches": matches,
            "predictions": predictions or [],
            "odds": odds_data or {}
        }

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"matches_{source}_{timestamp}.json"

        output_path = self.output_dir / output_file
        self._save_json(export_data, output_path)
        self._create_index_entry(output_path, export_data)
        return export_data

    def _save_json(self, data: Dict[str, Any], output_path: Path) -> None:
        """Save data to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Saved JSON export to: {output_path}")

    def _create_index_entry(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Create/update index file for data retrieval"""
        index_path = self.output_dir / "data_index.json"
        index = {}
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    index = json.load(f)
            except Exception:
                pass

        entry = {
            "file": str(file_path.name),
            "path": str(file_path),
            "created_at": data.get("metadata", {}).get("created_at", ""),
            "total_matches": data.get("metadata", {}).get("total_matches", 0),
            "source": data.get("metadata", {}).get("source", "")
        }
        index[file_path.name] = entry

        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)

    def export_to_csv(self, data: Dict[str, Any], filepath: str) -> str:
        """Export data to CSV format for spreadsheet analysis."""
        matches = data.get("matches", [])
        if not matches:
            logger.warning("No matches to export to CSV")
            return ""

        rows = [self._flatten_match_data(m) for m in matches]

        if rows:
            fieldnames = list(rows[0].keys())
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Exported {len(rows)} matches to CSV: {filepath}")

        return filepath

    def _flatten_match_data(self, match: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested match data for CSV export"""
        flattened = {}
        for key in ['home_team', 'away_team', 'league', 'sport', 'status', 'match_id', 'venue']:
            flattened[key] = match.get(key, '')

        flattened['start_time'] = match.get('start_time', '')

        odds = match.get('odds', {})
        flattened['odds_home'] = odds.get('home', odds.get('h2h', {}).get('home', 0))
        flattened['odds_draw'] = odds.get('draw', odds.get('h2h', {}).get('draw', 0))
        flattened['odds_away'] = odds.get('away', odds.get('h2h', {}).get('away', 0))
        flattened['odds_btts_yes'] = odds.get('btts', {}).get('yes', 0)
        flattened['odds_btts_no'] = odds.get('btts', {}).get('no', 0)
        flattened['odds_over_2_5'] = odds.get('over_under', {}).get('over_2_5', 0)
        flattened['odds_under_2_5'] = odds.get('over_under', {}).get('under_2_5', 0)

        for key in ['home_xG', 'away_xG', 'home_xGA', 'away_xGA', 'home_xGD', 'away_xGD']:
            flattened[key] = match.get(key, 0)

        if 'prediction' in match:
            pred = match['prediction']
            flattened['predicted_outcome'] = pred.get('outcome', '')
            flattened['model_probability'] = pred.get('probability', pred.get('model_probability', 0))
            flattened['value_score'] = pred.get('value', pred.get('value_score', 0))

        return flattened

    def export_to_excel(
        self, data: Dict[str, Any], filepath: str
    ) -> str:
        """Export data to Excel format with multiple sheets."""
        try:
            import pandas as pd

            sheets = {}

            matches = data.get("matches", [])
            if matches:
                sheets["matches"] = pd.DataFrame([self._flatten_match_data(m) for m in matches])

            predictions = data.get("predictions", [])
            if predictions:
                pred_rows = []
                for p in predictions:
                    match_info = p.get("match", {})
                    pred_row = {
                        **p.get("prediction", {}),
                        "match": f"{match_info.get('home_team', '')} vs {match_info.get('away_team', '')}"
                    }
                    pred_rows.append(pred_row)
                sheets["predictions"] = pd.DataFrame(pred_rows)

            value_bets = data.get("value_bets", [])
            if value_bets:
                sheets["value_bets"] = pd.DataFrame(value_bets)

            metadata = data.get("metadata", {})
            if metadata:
                sheets["metadata"] = pd.DataFrame([metadata])

            if sheets:
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    for sheet_name, df in sheets.items():
                        if df is not None and not df.empty:
                            df.to_excel(writer, sheet_name=sheet_name, index=False)

                logger.info(f"Exported to Excel: {filepath}")
                return filepath

        except ImportError:
            logger.error("pandas not available for Excel export")
            warnings.warn("pandas not installed - Excel export unavailable", UserWarning)
            return ""

    def generate_documentation(
        self, data: Dict[str, Any],
        filepath: Optional[str] = None,
        title: str = "Sports Data Export Documentation"
    ) -> str:
        """Generate markdown documentation from data."""
        metadata = data.get("metadata", {})
        matches = data.get("matches", [])
        predictions = data.get("predictions", [])
        value_bets = data.get("value_bets", [])
        extraction = data.get("extraction", {})

        doc = [f"# {title}\n"]
        doc.append(f"Generated: {metadata.get('created_at', datetime.now().isoformat())}\n")
        doc.append(f"Schema Version: {metadata.get('schema_version', '1.0')}\n\n")

        if metadata.get("source"):
            doc.append(f"## Source\n")
            doc.append(f"- **Source**: {metadata.get('source')}\n\n")

        doc.append(f"## Summary\n")
        doc.append(f"- Total Matches: {len(matches)}\n")
        doc.append(f"- Matches Analyzed: {len(predictions)}\n")
        doc.append(f"- Value Bets Found: {len(value_bets)}\n\n")

        doc.append("## Data Schema\n\n")
        doc.append("```json\n")
        doc.append('{"matches": [MatchSchema], "predictions": [PredictionResultSchema], "odds": OddsSchema, "metadata": MetadataSchema}\n')
        doc.append("```\n")

        if matches:
            doc.append(f"\n## Matches Extracted ({len(matches)})\n\n")
            doc.append("| # | Home Team | Away Team | Odds Home | Odds Draw | Odds Away |\n")
            doc.append("|---|-----------|-----------|-----------|-----------|----------|\n")

            for i, match in enumerate(matches[:20], 1):
                odds = match.get('odds', {})
                h2h = odds.get('h2h', odds) if isinstance(odds, dict) else odds
                doc.append(f"| {i} | {match.get('home_team', 'Unknown')} | {match.get('away_team', 'Unknown')} | {h2h.get('home', 0):.2f} | {h2h.get('draw', 0):.2f} | {h2h.get('away', 0):.2f} |\n")

            if len(matches) > 20:
                doc.append(f"\n*... and {len(matches) - 20} more matches*\n")

        if value_bets:
            doc.append(f"\n## Value Bets\n\n")
            doc.append("| Match | Outcome | Value Score | Model Probability |\n")
            doc.append("|-------|---------|-------------|-------------------|\n")

            sorted_bets = sorted(value_bets, key=lambda x: abs(x.get('value', 0)), reverse=True)
            for bet in sorted_bets[:10]:
                match = bet.get('match', '')
                outcome = bet.get('outcome', '')
                value = bet.get('value', 0)
                prob = bet.get('model_probability', 0)
                doc.append(f"| {match} | {outcome} | {value:.4f} | {prob:.2%} |\n")

        doc.append(f"\n## Data Lineage\n\n")
        doc.append("```\n")
        doc.append("SportyBet Code → extraction → analysis → prediction → export\n")
        doc.append("```\n")

        doc_str = "".join(doc)

        if filepath:
            output_file = Path(filepath)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(doc_str)
            logger.info(f"Documentation saved to: {filepath}")

        return doc_str

    def create_data_archive(
        self, data: Dict[str, Any],
        output_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        compression: str = "gzip"
    ) -> Path:
        """Create a complete data archive with JSON, CSV, and documentation."""
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        archive_dir = output_dir / name
        archive_dir.mkdir(parents=True, exist_ok=True)

        json_path = archive_dir / "data.json"
        self._save_json(data, json_path)

        csv_path = archive_dir / "matches.csv"
        self.export_to_csv(data, str(csv_path))

        doc_path = archive_dir / "documentation.md"
        self.generate_documentation(data, str(doc_path))

        index_path = archive_dir / "index.json"
        archive_index = {
            "archive_id": name,
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "files": {
                "json": "data.json",
                "csv": "matches.csv",
                "documentation": "documentation.md"
            },
            "total_matches": len(data.get("matches", [])),
            "total_predictions": len(data.get("predictions", [])),
            "total_value_bets": len(data.get("value_bets", []))
        }
        with open(index_path, 'w') as f:
            json.dump(archive_index, f, indent=2)

        logger.info(f"Created data archive: {archive_dir}")
        return archive_dir

    def cleanup_old_archives(
        self, days_to_keep: int = 30, dry_run: bool = False
    ) -> List[Path]:
        """Clean up old archive files according to retention policy."""
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted = []

        for archive_dir in self.output_dir.glob("export_*"):
            if archive_dir.is_dir():
                mtime = datetime.fromtimestamp(archive_dir.stat().st_mtime)
                if mtime < cutoff_date:
                    if dry_run:
                        deleted.append(archive_dir)
                    else:
                        import shutil
                        shutil.rmtree(archive_dir)
                        deleted.append(archive_dir)

        return deleted


# =============================================================================
# Standalone Functions
# =============================================================================

def export_matches_to_json(
    matches: List[Dict],
    output_path: str,
    metadata: Optional[Dict] = None,
    predictions: Optional[List[Dict]] = None,
    odds_data: Optional[Dict] = None,
    source: str = "sportybet"
) -> str:
    """Convenience function to export matches to JSON."""
    exporter = JSONExporter()
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "matches": matches,
        "predictions": predictions or [],
        "odds": odds_data or {},
        "metadata": metadata or {
            "exported_at": now,
            "run_id": exporter._run_id,
            "source": source,
            "total_matches": len(matches)
        },
        "extraction": {"start_time": now, "end_time": now},
        "analysis": {"start_time": now, "end_time": now}
    }
    return exporter.export_matches_to_json(
        matches, predictions, odds_data, source, Path(output_path).name
    )


def create_data_archive(
    matches: List[Dict],
    predictions: Optional[List[Dict]] = None,
    output_dir: str = "output",
    code: Optional[str] = None,
    compression: str = "gzip"
) -> Path:
    """Create a complete data archive."""
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "matches": matches,
        "predictions": predictions or [],
        "metadata": {"exported_at": now, "code": code},
        "extraction": {"start_time": now, "end_time": now},
        "analysis": {"start_time": now, "end_time": now}
    }

    exporter = JSONExporter(output_dir)
    return exporter.create_data_archive(data, compression=compression)


def generate_documentation(
    data: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "Sports Data Export Documentation"
) -> str:
    """Generate markdown documentation from data."""
    exporter = JSONExporter()
    return exporter.generate_documentation(data, output_path, title)


def create_data_index(
    data_dir: str,
    index_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create an index file for easy data retrieval."""
    data_dir_path = Path(data_dir)

    if index_path is None:
        index_path = data_dir_path / "data_index.json"
    else:
        index_path = Path(index_path)

    index = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(data_dir_path),
        "files": {}
    }

    for json_file in data_dir_path.glob("*.json"):
        if json_file.name == "data_index.json":
            continue

        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)

            stat = json_file.stat()

            index["files"][json_file.name] = {
                "path": str(json_file),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "total_matches": file_data.get("metadata", {}).get("total_matches", 0)
            }
        except Exception as e:
            logger.warning(f"Could not index {json_file}: {e}")

    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)

    return index


def organize_by_date_sport(
    source_dir: str,
    output_base: str = "./organized_data"
) -> Dict[str, Path]:
    """Organize data files by date and sport."""
    source_path = Path(source_dir)
    output_base_path = Path(output_base)
    output_base_path.mkdir(parents=True, exist_ok=True)

    organized_paths = {}

    for json_file in source_path.glob("*.json"):
        if json_file.name == "data_index.json":
            continue

        try:
            with open(json_file, 'r') as f:
                file_data = json.load(f)

            created_at = file_data.get("metadata", {}).get("created_at", "")
            if not created_at:
                created_at = datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()

            date_str = created_at.split("T")[0] if isinstance(created_at, str) else datetime.now().strftime("%Y-%m-%d")

            sport = "football"
            if file_data.get("matches"):
                sport = file_data["matches"][0].get("sport", "football")
            elif file_data.get("metadata", {}).get("sport"):
                sport = file_data["metadata"]["sport"]

            sport_dir = output_base_path / sport
            sport_dir.mkdir(exist_ok=True)

            new_name = f"{json_file.stem}_{date_str}.json"
            new_path = sport_dir / new_name

            import shutil
            shutil.copy2(json_file, new_path)
            organized_paths[json_file.name] = new_path

        except Exception as e:
            logger.error(f"Error organizing {json_file}: {e}")

    return organized_paths


# =============================================================================
# CLI Entry Point
# =============================================================================


if __name__ == "__main__":
    # Demo usage
    demo_matches = [{
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "league": "Premier League",
        "sport": "football",
        "odds": {"h2h": {"home": 2.1, "draw": 3.5, "away": 4.0}},
        "home_xG": 1.8, "away_xG": 1.2
    }]

    # Predictions are flat dictionaries matching PredictionResultSchema
    demo_predictions = [{
        "match": "Arsenal vs Chelsea",
        "outcome": "Home",
        "value": 0.15,
        "model_probability": 0.65,
        "implied_probability": 0.48,
        "is_value_bet": True,
        "confidence_level": "high"
    }]

    exporter = JSONExporter("output")
    result = exporter.export_matches_to_json(demo_matches, demo_predictions, source="demo")
    print(f"Exported: {result['metadata']['run_id']}")
    doc = exporter.generate_documentation(result)
    print(f"Documentation generated: {len(doc)} characters")
    print("JSON Exporter module loaded and tested successfully!")
