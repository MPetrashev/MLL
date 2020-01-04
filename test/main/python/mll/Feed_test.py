import unittest
import datetime
import quandl


class Feed_test(unittest.TestCase):
    def test_oil_feed(self):
        df = quandl.get("CHRIS/CME_CL8")
        self.assertGreater(df.index[-1], datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'))


if __name__ == '__main__':
    unittest.main()
