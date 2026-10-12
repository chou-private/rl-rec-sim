import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('calibration', ROOT / 'src/core/envs/Simulated_Env/yahoo_calibration.py')
calibration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calibration)


class CalibrationTests(unittest.TestCase):
    def test_reward_scale_and_bounds(self):
        result = calibration.calibrated_reward(np.array([-3, 3, 8]), np.array([0, .5, 1]), 1)
        np.testing.assert_allclose(result, [0, 0, 0])
        np.testing.assert_allclose(calibration.calibrated_reward(np.array([1, 3, 5]), 0, 0), [0, .5, 1])
        with self.assertRaises(ValueError):
            calibration.calibrated_reward(3, 0, -1)

    def test_basis_interpolation(self):
        x = calibration.design([1, 2.5, 5], np.zeros((3, 0)))
        np.testing.assert_allclose(x.sum(axis=1), 1)
        np.testing.assert_allclose(x[1], [0, .5, .5, 0, 0])

    def test_artifact_identity_and_disjoint_users(self):
        pred = np.ones((3, 2))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'artifact.npz'
            np.savez(path, prediction_hash=calibration.matrix_hash(pred), fit_users=[0], test_users=[2], risk_survey=pred * .2)
            risk, train, test = calibration.load_artifact(path, pred, 'survey')
            np.testing.assert_allclose(risk, .2)
            with self.assertRaises(ValueError):
                calibration.load_artifact(path, pred + 1, 'survey')


if __name__ == '__main__':
    unittest.main()
