# Refactored Architecture Plan

## Current State Analysis

### Existing Components
1. **main.py** - Telegram bot (TO BE REMOVED)
2. **extractor.py** - Playwright SportyBet scraping (REFURBISH)
3. **full_data_pipeline.py** - API sports data pipeline (REFURBISH)
4. **v3_prediction_model.py** - ML prediction model (ENHANCE)
5. **bet_analyzer.py** - Standalone CLI analyzer (UNIFIED INTO MAIN)
6. **verify_creds.py** - Credential checker (REFERENCE)

## Target Architecture

```
sport_predictor/
├── src/
│   ├── data_extractor/           # Web scraping module
│   │   ├── __init__.py
│   │   ├── sportybet_scraper.py  # Playwright-based extraction
│   │   └── scraper_utils.py       # Helper functions
│   ├── feature_engineering/      # ML feature preparation
│   │   ├── __init__.py
│   │   ├── feature_builder.py
│   │   └── feature_scaler.py
│   ├── models/                   # ML models & prediction
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── model_trainer.py
│   │   └── model_registry.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_validator.py
│   │   └── config.py
│   └── cli.py                    # Main CLI entry point
├── tests/
│   ├── conftest.py
│   └── test_*.py
├── .env.example
└── requirements.txt
```

## Implementation Steps

### Phase 1: Remove Telegram Dependency
1. Create new `cli.py` as main entry point
2. Remove Telegram bot commands from `main.py` (or deprecate entirely)
3. Update all functions to work without Telegram context

### Phase 2: Enhanced Web Scraping
1. Refurbish `extractor.py` → `sportybet_scraper.py`
2. Add proper error handling and retry logic
3. Implement structured data extraction with fallback strategies
4. Add logging and progress reporting

### Phase 3: Remove Odds API
1. Replace `fetch_odds()` in `full_data_pipeline.py` with local odds calculation
2. Keep API-Sports for fixtures/statistics, remove OddsAPI dependency
3. Calculate implied probabilities from extracted odds only

### Phase 4: Machine Learning Integration
1. Enhance `v3_prediction_model.py`:
   - Replace mock model with real training pipeline
   - Add model versioning
   - Implement model evaluation metrics
   - Add feature importance analysis

2. Create model training utilities:
   - Data preparation from historical matches
   - Cross-validation
   - Hyperparameter tuning
   - Model persistence

### Phase 5: AI Integration
1. Add neural network alternatives to XGBoost:
   - Simple PyTorch/TensorFlow model
   - LSTM for temporal patterns
   - Attention mechanisms for feature weighting

2. Implement ensemble predictions:
   - Combine multiple models
   - Bayesian model averaging
   - Uncertainty quantification

### Phase 6: Cleaning & Refactoring
1. Code review and cleanup
2. Remove duplicate code
3. Add proper type hints
4. Add comprehensive docstrings
5. Create tests for all new modules

## Key Changes

### From Current to Target:

| Current | Target | Change |
|---------|--------|--------|
| Telegram bot | CLI tool | Remove instant messaging dependency |
| OddsAPI | Local odds | Remove external API for market odds |
| Mock model | Real ML | Replace random predictions with trained models |
| Single file | Modular structure | Better organization and maintainability |
| Hardcoded values | Config-driven | More flexible and testable |