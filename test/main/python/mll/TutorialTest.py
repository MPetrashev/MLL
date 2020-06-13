import unittest
import numpy as np

from mll.Tutorial import Tutorial


class TutorialTest(unittest.TestCase):
    # unit test prediction LSTM
    def test_prediction(self):
        t = Tutorial()
        prediction = [p[0][0] for p in t.predict(3)]
        np.testing.assert_almost_equal([0.5958766, 0.6530068, 1.3395267], prediction, decimal=7) # 0.5945736, 0.65164703, 1.3375353 value from https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb


if __name__ == '__main__':
    unittest.main()
