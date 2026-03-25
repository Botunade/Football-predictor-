import unittest
from main import determine_status
from datetime import datetime, timezone, timedelta

class TestSyncLogic(unittest.TestCase):
    def test_upcoming(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.assertEqual(determine_status(future), "upcoming")

    def test_running(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        # default duration is 2h, so it should be running
        self.assertEqual(determine_status(past), "running")

    def test_finished(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        self.assertEqual(determine_status(old), "finished")

if __name__ == "__main__":
    unittest.main()
