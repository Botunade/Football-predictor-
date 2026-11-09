import unittest
from unittest.mock import patch, MagicMock
from src.data_extractor.stats_scrapers.multisport_scraper import MultiSportScraper


class TestMultiSportScraper(unittest.TestCase):
    def test_multisport_basketball_structure(self):
        scraper = MultiSportScraper()
        # Mock API response for basketball
        mock_payload = {
            "data": {
                "tournaments": [
                    {
                        "name": "NBA",
                        "events": [
                            {
                                "homeTeamName": "Boston Celtics",
                                "awayTeamName": "LA Lakers",
                                "eventId": "sr:match:123456",
                                "markets": [
                                    {
                                        "id": "1",
                                        "outcomes": [
                                            {"id": "1", "odds": "1.45"},
                                            {"id": "2", "odds": "2.80"}
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        with patch("urllib.request.urlopen") as mock_url:
            mock_resp = MagicMock()
            import json
            mock_resp.read.return_value = json.dumps(mock_payload).encode("utf-8")
            mock_url.return_value.__enter__.return_value = mock_resp

            fixtures = scraper.fetch_basketball_fixtures()
            self.assertEqual(len(fixtures), 1)
            self.assertEqual(fixtures[0]["sport"], "basketball")
            self.assertEqual(fixtures[0]["home_team"], "Boston Celtics")
            self.assertEqual(fixtures[0]["odds_home"], 1.45)
            self.assertEqual(fixtures[0]["odds_away"], 2.80)
            self.assertEqual(fixtures[0]["recommended_outcome"], "Home")


if __name__ == "__main__":
    unittest.main()
