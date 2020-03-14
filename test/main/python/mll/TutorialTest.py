import unittest

from mll.Tutorial import Tutorial


class TutorialTest(unittest.TestCase):
    # unit test prediction LSTM
    def test_prediction(self):
        t = Tutorial()
        prediction = t.predict()
        self.assertAlmostEqual(0.60141844, prediction[0][0][0], places=6)


if __name__ == '__main__':
    unittest.main()
