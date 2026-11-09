import unittest
from src.data_extractor.stats_scrapers.sportybet_league_spider import SportyBetLeagueSpider


class TestLeagueSpider(unittest.TestCase):
    def test_spider_parsing_structure(self):
        spider = SportyBetLeagueSpider()
        # Verify parser helper correctly decodes a mock SportyBet event with markets
        mock_event = {
            "eventId": "sr:match:999999",
            "homeTeamName": "Sunderland AFC",
            "awayTeamName": "Chelsea",
            "markets": [
                {
                    "id": "1",
                    "desc": "1X2",
                    "outcomes": [
                        {"id": "1", "desc": "Home", "odds": "3.95"},
                        {"id": "2", "desc": "Draw", "odds": "3.82"},
                        {"id": "3", "desc": "Away", "odds": "1.98"}
                    ]
                },
                {
                    "id": "18",
                    "desc": "Over/Under",
                    "specifier": "total=1.5",
                    "outcomes": [
                        {"id": "12", "desc": "Over", "odds": "1.25"},
                        {"id": "13", "desc": "Under", "odds": "3.80"}
                    ]
                }
            ]
        }

        oh, od, oa, m_odds = spider.scraper_helper._parse_event_markets(mock_event)
        self.assertEqual(oh, 3.95)
        self.assertEqual(od, 3.82)
        self.assertEqual(oa, 1.98)
        self.assertEqual(m_odds.get("Home"), 3.95)
        self.assertEqual(m_odds.get("Away"), 1.98)
        self.assertEqual(m_odds.get("Over 1.5"), 1.25)


if __name__ == "__main__":
    unittest.main()
