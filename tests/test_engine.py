from __future__ import annotations

import unittest

import numpy as np

from edvf_rss import DispatchConfig, edvf_flow, simulate_independent_day


class EDVFEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = DispatchConfig(fleet=2, response_slots=4, horizon=3, scenarios=1)
        self.distance = np.array(
            [[0.0, 10.0, 20.0], [10.0, 0.0, 10.0], [20.0, 10.0, 0.0]]
        )

    def test_flow_preserves_fleet_and_relocation_limit(self) -> None:
        current = np.array([2, 0, 0])
        visible = np.zeros((4, 3))
        visible[-1, 1] = 35.0
        flow = edvf_flow(current, visible, [], self.distance, self.cfg).first_flow
        np.testing.assert_array_equal(flow.sum(axis=1), current)
        self.assertLessEqual(int(flow.sum() - np.trace(flow)), 1)

    def test_day_replay_conserves_rss(self) -> None:
        actual = np.zeros((48, 3))
        actual[2, 0] = 10.0
        actual[4, 1] = 30.0
        actual[8, 2] = 15.0
        history = np.stack([actual])
        result = simulate_independent_day(
            actual,
            actual,
            history,
            np.array([0]),
            self.distance,
            self.cfg,
            "EDVF-RSS-Exact",
            include_future=False,
        )
        self.assertGreater(result.total_rss_kwh, 0.0)
        self.assertAlmostEqual(
            result.total_rss_kwh,
            result.recovered_rss_kwh + result.residual_rss_kwh,
            places=7,
        )
        self.assertLessEqual(result.max_conservation_error, 1e-6)


if __name__ == "__main__":
    unittest.main()
