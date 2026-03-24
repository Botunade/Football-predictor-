import unittest
from extractor import _parse_raw_text

class TestExtractorParsing(unittest.TestCase):
    def test_football_parse(self):
        raw = "Arsenal vs Chelsea\n1X2\n1.85\n3.20\n4.50"
        parsed = _parse_raw_text(raw)
        self.assertEqual(parsed["home"], "Arsenal")
        self.assertEqual(parsed["away"], "Chelsea")
        self.assertEqual(parsed["odds_home"], 1.85)
        self.assertEqual(parsed["odds_draw"], 3.20)
        self.assertEqual(parsed["odds_away"], 4.50)
        self.assertEqual(parsed["sport"], "football")

    def test_basketball_parse(self):
        raw = "Lakers vs Celtics\nHome/Away\n1.90\n1.95"
        parsed = _parse_raw_text(raw)
        self.assertEqual(parsed["home"], "Lakers")
        self.assertEqual(parsed["away"], "Celtics")
        self.assertEqual(parsed["odds_home"], 1.90)
        self.assertIsNone(parsed["odds_draw"])
        self.assertEqual(parsed["odds_away"], 1.95)
        self.assertEqual(parsed["sport"], "basketball")

    def test_regex_fallback(self):
        raw = "Premier League Match: Man Utd vs Liverpool scheduled for tomorrow"
        parsed = _parse_raw_text(raw)
        self.assertEqual(parsed["home"], "Man Utd")
        self.assertEqual(parsed["away"], "Liverpool")

if __name__ == "__main__":
    unittest.main()
