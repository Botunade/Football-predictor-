import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from main import handle_manual_input

async def test_multi_sport_input_parsing():
    # Mock Update and Context
    mock_update = MagicMock()
    mock_update.message.reply_text = AsyncMock()
    mock_update.effective_chat.id = 7104386905

    # Ensure current environment CHAT_ID matches
    import main
    main.CHAT_ID = "7104386905"

    # 1. Test Basketball
    mock_update.message.text = "Basketball | Lakers vs Celtics | 1.90 | 1.95"
    with patch('full_data_pipeline.fetch_player_info') as mock_player, \
         patch('v3_prediction_model.predict_match') as mock_predict, \
         patch('main.format_telegram_message') as mock_format:

        mock_player.return_value = {"injury_index": 0.1, "fatigue_index": 0.2, "key_player_dependency": 0.5}
        mock_predict.return_value = {"h2h": {"model_probability": 0.60, "implied_probability": 0.51, "value": 0.09, "outcome": "Home"}}
        mock_format.return_value = "Test Basketball Message"

        await handle_manual_input(mock_update, mock_context := MagicMock())

        self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
        print(f"Basketball replies: {self_replies}")
        assert any("Lakers" in r and "Basketball" in r for r in self_replies)
        assert "Test Basketball Message" in self_replies

    # 2. Test Hockey
    mock_update.message.reply_text.reset_mock()
    mock_update.message.text = "Hockey | Rangers vs Penguins | 2.10 | 1.80"
    with patch('full_data_pipeline.fetch_player_info') as mock_player, \
         patch('v3_prediction_model.predict_match') as mock_predict, \
         patch('main.format_telegram_message') as mock_format:

        mock_player.return_value = {"injury_index": 0.1, "fatigue_index": 0.2, "key_player_dependency": 0.5}
        mock_predict.return_value = {"h2h": {"model_probability": 0.60, "implied_probability": 0.47, "value": 0.13, "outcome": "Home"}}
        mock_format.return_value = "Test Hockey Message"

        await handle_manual_input(mock_update, mock_context := MagicMock())

        self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
        print(f"Hockey replies: {self_replies}")
        assert any("Rangers" in r and "Hockey" in r for r in self_replies)
        assert "Test Hockey Message" in self_replies

    # 3. Test Invalid Sport
    mock_update.message.reply_text.reset_mock()
    mock_update.message.text = "Tennis | Nadal vs Federer | 1.50 | 2.50"
    await handle_manual_input(mock_update, mock_context := MagicMock())
    self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
    assert any("Sport 'tennis' not supported" in r for r in self_replies)

    print("Multi-sport input tests passed!")

if __name__ == "__main__":
    import os
    os.environ["CHAT_ID"] = "7104386905"
    asyncio.run(test_multi_sport_input_parsing())
