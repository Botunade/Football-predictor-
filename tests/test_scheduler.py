import unittest
from src.automation.scheduler import AutonomousScheduler, JobConfig


class TestAutonomousScheduler(unittest.TestCase):
    def test_scheduler_lifecycle_and_config(self):
        cfg = JobConfig(
            enabled=True,
            interval_hours=12,
            target_odds=4.0,
            stake=100.0,
            strategy="max_prob"
        )
        scheduler = AutonomousScheduler(config=cfg)
        status = scheduler.get_status()
        self.assertTrue(status["enabled"])
        self.assertEqual(status["target_odds"], 4.0)
        self.assertEqual(status["stake"], 100.0)

        # Update config dynamically
        scheduler.update_config(target_odds=5.5, stake=200.0)
        new_status = scheduler.get_status()
        self.assertEqual(new_status["target_odds"], 5.5)
        self.assertEqual(new_status["stake"], 200.0)


if __name__ == "__main__":
    unittest.main()
