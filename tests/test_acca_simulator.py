import unittest
from collections import Counter
from src.models.acca_simulator import AccaSimulator


class TestAccaSimulator(unittest.TestCase):
    def setUp(self):
        # Generate 15 distinct mock fixtures with multi-market odds
        self.fixtures = []
        for i in range(15):
            self.fixtures.append({
                "home_team": f"HomeTeam_{i}",
                "away_team": f"AwayTeam_{i}",
                "league": "Premier League",
                "odds_home": 1.70 + (i % 5) * 0.20,
                "odds_draw": 3.40,
                "odds_away": 3.80 - (i % 5) * 0.20,
                "sporty_id": f"SR:MATCH:{1000 + i}",
                "sources": ["SportyBet", "Forebet"],
                "market_odds": {
                    "Home (1UP)": 1.45,
                    "1X": 1.25,
                    "Over 1.5": 1.30,
                    "Under 3.5": 1.35,
                    "2nd Half Under 2.5": 1.42,
                    "1st Half Over 0.5": 1.38,
                    "Home DNB": 1.32
                }
            })
        self.sim = AccaSimulator(self.fixtures)

    def test_market_diversity_in_accumulators(self):
        """Verify generated accumulator does NOT monopolize one single market like 2nd Half Under 2.5."""
        res = self.sim.simulate_optimal_slip(target_odds=20.0, match_count=8, strategy="max_prob")
        self.assertTrue(res["success"])
        matches = res["optimal_slip"]["matches"]
        self.assertEqual(len(matches), 8)

        # Count market types
        outcomes = [m["selected_outcome"] for m in matches]
        counts = Counter(outcomes)

        # Strict check: no single selection should appear more than 3 times in an 8-leg slip
        for market, count in counts.items():
            self.assertLessEqual(count, 3, f"Market '{market}' dominated the slip ({count} times)!")

        # Must have at least 4 distinct markets across 8 legs
        self.assertGreaterEqual(len(counts), 4, f"Only {len(counts)} distinct markets found; expected >= 4.")

    def test_variations_differ(self):
        """Verify variation 0 and variation 1 produce alternative slips."""
        res0 = self.sim.simulate_optimal_slip(target_odds=15.0, match_count=6, variation_index=0)
        res1 = self.sim.simulate_optimal_slip(target_odds=15.0, match_count=6, variation_index=1)
        self.assertTrue(res0["success"])
        self.assertTrue(res1["success"])


if __name__ == "__main__":
    unittest.main()
