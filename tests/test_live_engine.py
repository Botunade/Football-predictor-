import unittest
from unittest.mock import patch
from src.models.live_engine import LiveBettingEngine, LiveMarketOption


class TestLiveBettingEngine(unittest.TestCase):
    def setUp(self):
        self.engine = LiveBettingEngine()

    def test_clock_and_score_parsing(self):
        ev = {
            "playedSeconds": "63:45",
            "matchStatus": "H2",
            "period": 2,
            "setScore": "1:2"
        }
        minute, clock_str, period_desc = self.engine._parse_match_clock(ev)
        h_score, a_score, score_str = self.engine._parse_score(ev)

        self.assertEqual(minute, 63)
        self.assertEqual(period_desc, "2nd Half")
        self.assertEqual(h_score, 1)
        self.assertEqual(a_score, 2)
        self.assertEqual(score_str, "1-2")

    def test_analyze_live_match_filters_settled_lines(self):
        # Match is 1-2 (total 3 goals) at 65'
        ev = {
            "eventId": "sr:match:999999",
            "homeTeamName": "Live Arsenal",
            "awayTeamName": "Live Chelsea",
            "playedSeconds": "65:00",
            "matchStatus": "H2",
            "period": 2,
            "setScore": "1:2",
            "markets": [
                {
                    "id": "18",
                    "desc": "Over/Under",
                    "specifier": "total=1.5",
                    "outcomes": [{"id": "12", "desc": "Over 1.5", "odds": "1.05", "isActive": 1}]
                },
                {
                    "id": "18",
                    "desc": "Over/Under",
                    "specifier": "total=3.5",
                    "outcomes": [
                        {"id": "12", "desc": "Over 3.5", "odds": "1.85", "isActive": 1},
                        {"id": "13", "desc": "Under 3.5", "odds": "1.95", "isActive": 1}
                    ]
                },
                {
                    "id": "1",
                    "desc": "1X2",
                    "specifier": "",
                    "outcomes": [
                        {"id": "1", "desc": "Home", "odds": "9.50", "isActive": 1},
                        {"id": "2", "desc": "Draw", "odds": "4.20", "isActive": 1},
                        {"id": "3", "desc": "Away", "odds": "1.25", "isActive": 1}
                    ]
                }
            ]
        }

        analysis = self.engine.analyze_live_match(ev)
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.home_score, 1)
        self.assertEqual(analysis.away_score, 2)

        # Over 1.5 should be filtered out because total goals is already 3
        option_names = [o.outcome_name for o in analysis.all_options]
        self.assertNotIn("Over 1.5", option_names)
        self.assertTrue(any("3.5" in o.outcome_name or o.outcome_name in ("Home", "Draw", "Away") for o in analysis.all_options))

    @patch("src.automation.sportybet_bot.SportyBetAutomator.generate_booking_code_via_api")
    def test_build_live_accumulator_slip(self, mock_gen_code):
        mock_gen_code.return_value = "LIVE77"

        selections = [
            {
                "sporty_id": "sr:match:111",
                "home_team": "Team A",
                "away_team": "Team B",
                "outcome_name": "Over 2.5",
                "odds": 1.60,
                "prob": 0.70,
                "market_id": "18",
                "outcome_id": "12",
                "specifier": "total=2.5"
            },
            {
                "sporty_id": "sr:match:222",
                "home_team": "Team C",
                "away_team": "Team D",
                "outcome_name": "Away",
                "odds": 1.40,
                "prob_pct": 75.0,
                "market_id": "1",
                "outcome_id": "3",
                "specifier": ""
            }
        ]

        res = self.engine.build_live_accumulator_slip(selections, stake=100.0)
        self.assertTrue(res["success"])
        self.assertEqual(res["booking_code"], "LIVE77")
        self.assertEqual(res["total_legs"], 2)
        self.assertEqual(res["combined_odds"], 2.24)
        self.assertEqual(res["potential_win"], 224.0)


if __name__ == "__main__":
    unittest.main()
