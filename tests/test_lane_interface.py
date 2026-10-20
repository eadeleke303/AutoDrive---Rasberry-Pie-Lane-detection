import unittest
from lane_interface import observation_from_boundaries

class LaneInterfaceTests(unittest.TestCase):
    def test_centered(self):
        obs=observation_from_boundaries(
            {"left_lane_boundary":[[360,300],[360,500]],"right_lane_boundary":[[600,300],[600,500]]},
            {"reason":"test","selected_channels":[1,2]},960,540)
        self.assertEqual(obs.status,"detected")
        self.assertAlmostEqual(obs.error_normalized,0)
    def test_right_offset_positive(self):
        obs=observation_from_boundaries(
            {"left_lane_boundary":[[400,300],[400,500]],"right_lane_boundary":[[640,300],[640,500]]},
            {"reason":"test"},960,540)
        self.assertGreater(obs.error_normalized,0)
    def test_missing_not_zero(self):
        obs=observation_from_boundaries({},{"reason":"no_pair"},960,540)
        self.assertEqual(obs.status,"missing")
        self.assertIsNone(obs.error_normalized)
    def test_estimate_not_confirmed(self):
        obs=observation_from_boundaries(
            {"left_lane_boundary":[[360,300],[360,500]],"right_lane_boundary":[[600,300],[600,500]]},
            {"estimated":True},960,540)
        self.assertEqual(obs.status,"estimated")

if __name__=="__main__":
    unittest.main()
