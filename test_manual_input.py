import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from main import handle_manual_input

async def test_manual_input_parsing():
    # Mock Update and Context
    mock_update = MagicMock()
    mock_update.message.reply_text = AsyncMock()
    mock_update.effective_chat.id = 7104386905
    # Ensure current environment CHAT_ID matches
    import main
    main.CHAT_ID = "7104386905"
    mock_update.message.text = "Football | Arsenal vs Chelsea | 2.10 | 3.50 | 3.20"
    mock_context = MagicMock()

    # Mock internal methods to avoid API calls during test
    with patch('full_data_pipeline.scrape_understat_team_async', new_callable=AsyncMock) as mock_scrape, \
         patch('full_data_pipeline.fetch_player_info', new_callable=AsyncMock) as mock_player, \
         patch('v3_prediction_model.predict_match') as mock_predict:

        mock_scrape.return_value = {"xG": 1.5, "xGA": 1.2, "xGD": 0.3, "NPxG": 1.4, "PPDA": 10.5, "possession": 50.0}
        mock_player.return_value = {"injury_index": 0.1, "fatigue_index": 0.2, "key_player_dependency": 0.5}
        mock_predict.return_value = {
            "h2h": {"match": "Arsenal vs Chelsea", "model_probability": 0.60, "implied_probability": 0.48, "value": 0.12, "outcome": "Home"},
            "btts": {}, "over_25": {}
        }

        await handle_manual_input(mock_update, mock_context)

        # Verify first reply was "Analyzing..."
        self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
        print(f"Captured self-replies: {self_replies}")
        assert any("Analyzing Arsenal vs Chelsea" in r for r in self_replies)
        print("Manual input test passed!")

async def test_unauthorized_access():
    mock_update = MagicMock()
    mock_update.message.reply_text = AsyncMock()
    mock_update.effective_chat.id = 12345678 # Unauthorized ID
    mock_context = MagicMock()

    await handle_manual_input(mock_update, mock_context)

    mock_update.message.reply_text.assert_not_called()
    print("Unauthorized access test passed!")

if __name__ == "__main__":
    import os
    # Set CHAT_ID in environment for test
    os.environ["CHAT_ID"] = "7104386905"
    asyncio.run(test_manual_input_parsing())
    asyncio.run(test_unauthorized_access())
