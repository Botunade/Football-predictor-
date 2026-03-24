import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from main_lite import handle_lite_input

async def test_lite_flow():
    # Mock Update and Context
    mock_update = MagicMock()
    mock_update.message.reply_text = AsyncMock()
    mock_update.effective_chat.id = 7104386905

    # Ensure current environment CHAT_ID matches
    import main_lite
    main_lite.CHAT_ID = "7104386905"

    # 1. Test CODE Input
    mock_update.message.text = "CODE | LITE123"
    with patch('main_lite.extract_booking_code_data_lite') as mock_extract:
        mock_extract.return_value = [
            {"home": "Team X", "away": "Team Y", "odds_home": 3.5, "odds_draw": 3.0, "odds_away": 2.0}
        ]

        await handle_lite_input(mock_update, MagicMock())

        self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
        print(f"Lite CODE replies: {self_replies}")
        assert any("Extracting" in r for r in self_replies)
        assert any("High-Value Bets" in r for r in self_replies)
        assert any("Team X vs Team Y" in r for r in self_replies)

    # 2. Test Manual Input (Lite)
    mock_update.message.reply_text.reset_mock()
    mock_update.message.text = "Football | Arsenal vs Chelsea | 2.5 | 3.2 | 2.8"

    await handle_lite_input(mock_update, MagicMock())
    self_replies = [call.args[0] for call in mock_update.message.reply_text.call_args_list]
    print(f"Lite Manual replies: {self_replies}")
    assert any("Analyzing" in r and "Arsenal" in r for r in self_replies)
    assert any("High-Value Bets" in r for r in self_replies)

    print("Lite flow tests passed!")

if __name__ == "__main__":
    import os
    os.environ["CHAT_ID"] = "7104386905"
    asyncio.run(test_lite_flow())
