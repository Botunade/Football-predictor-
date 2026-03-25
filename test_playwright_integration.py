import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from main import handle_sporty_code
from main import handle_manual_input

async def test_booking_code_integration():
    # Mock Update and Context
    mock_update = MagicMock()
    mock_update.message.reply_text = AsyncMock()
    mock_update.effective_chat.id = 7104386905

    # Ensure current environment CHAT_ID matches
    import main
    main.CHAT_ID = "7104386905"

    # Mock extractor, predictor, and feature builder
    with patch('main.extract_sporty_code') as mock_extract, \
         patch('main.background_analysis') as mock_background, \
         patch('main.format_matches_with_status_and_odds') as mock_format:

        mock_extract.return_value = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "sport": "football", "start_time": "2024-01-01T12:00:00Z", "odds_home": 1.8, "odds_draw": 3.4, "odds_away": 4.5}
        ]
        mock_format.return_value = "High-Value Bets from Code"

        mock_context = MagicMock()
        mock_context.args = ["BC123XYZ"]

        await handle_sporty_code(mock_update, mock_context)
    mock_update.message.text = "CODE | BC123XYZ"

    # Mock extractor, predictor, and feature builder
    with patch('main.extract_booking_code_data') as mock_extract, \
         patch('main.build_features') as mock_features, \
         patch('main.predict_match') as mock_predict, \
         patch('main.format_telegram_message') as mock_format:

        mock_extract.return_value = [
            {"home": "Arsenal", "away": "Chelsea", "sport": "football", "market": "1X2", "odds_home": 1.8, "odds_draw": 3.4, "odds_away": 4.5}
        ]
        mock_features.return_value = {"home_team": "Arsenal", "away_team": "Chelsea"}
        mock_predict.return_value = {"h2h": {"model_probability": 0.65, "implied_probability": 0.55, "value": 0.10, "outcome": "Home", "betting_code": "ABC"}}
        mock_format.return_value = "High-Value Bets from Code"

        await handle_manual_input(mock_update, MagicMock())

        self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
        print(f"Booking code replies: {self_replies}")
        assert any("Extracting matches" in r for r in self_replies)
        assert any("High-Value Bets from Code" in r for r in self_replies)

    print("Booking code integration test passed!")

if __name__ == "__main__":
    asyncio.run(test_booking_code_integration())
