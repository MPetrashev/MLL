import unittest

from mll.Tutorial import Tutorial


class Tutorial_test(unittest.TestCase):

    def test_prediction(self):
        t = Tutorial()
        prediction = t.predict()
        self.assertAlmostEqual(0.60139096, prediction[0][0][0], places=6)


if __name__ == '__main__':
    unittest.main()
