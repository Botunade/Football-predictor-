import unittest
from src.automation.notifier import Notifier


class TestNotifier(unittest.TestCase):
    def test_format_and_whatsapp_link(self):
        notifier = Notifier()
        matches = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "selection": "1st Half Over 0.5", "odds": 1.45},
            {"home_team": "Inter", "away_team": "Milan", "selection": "Home (1UP)", "odds": 1.35}
        ]
        msg = notifier.format_slip_message(
            booking_code="KXSTUG",
            combined_odds=1.96,
            stake=100.0,
            matches=matches
        )
        self.assertIn("KXSTUG", msg)
        self.assertIn("Arsenal vs Chelsea", msg)
        self.assertIn("Inter vs Milan", msg)

        # Test WhatsApp Click-to-Share URL generation
        wa_url = notifier.generate_whatsapp_share_url(msg)
        self.assertTrue(wa_url.startswith("https://api.whatsapp.com/send?text="))
        self.assertIn("KXSTUG", wa_url)

        # Test safe unconfigured telegram fallback
        res = notifier.send_telegram_message(msg)
        self.assertFalse(res["sent"])
        self.assertIn("whatsapp_url", res)


if __name__ == "__main__":
    unittest.main()
