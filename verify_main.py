import asyncio
import os
from main import run_version3_pipeline
from unittest.mock import patch, MagicMock

async def test_main_flow():
    # Mock API responses to avoid needing real keys
    with patch('main.fetch_fixtures') as mock_fix, \
         patch('main.fetch_odds') as mock_odds, \
         patch('main.send_telegram_message') as mock_send:

        mock_fix.return_value = [{
            "fixture": {"id": 1},
            "teams": {
                "home": {"id": 10, "name": "Team A"},
                "away": {"id": 20, "name": "Team B"}
            },
            "league": {"id": 39, "season": 2024}
        }]

        # High value odds
        mock_odds.return_value = [{
            "home_team": "Team A",
            "away_team": "Team B",
            "odds_home": 10.0, # Very high odds to ensure value > 0.08 if model prob is reasonable
            "odds_away": 1.1,
            "odds_draw": 3.0
        }]

        # Force a high model probability for Team A
        with patch('v3_prediction_model.model.predict_proba') as mock_predict:
            import numpy as np
            mock_predict.return_value = np.array([[0.9, 0.05, 0.05]])

            await run_version3_pipeline(39, 2024)

            # Verify message was sent
            mock_send.assert_called()
            args, _ = mock_send.call_args
            print("Captured Telegram Message:")
            print(args[0])
            assert "High-Confidence Value Bets" in args[0]
            assert "Team A vs Team B" in args[0]
            assert "H2H" in args[0]

if __name__ == "__main__":
    asyncio.run(test_main_flow())
