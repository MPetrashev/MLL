import unittest
import datetime
import quandl
import mll.QuandlFeed as qf
from TestCase import TestCase


class QuandlFeedTest(TestCase):
    def test_oil_feed(self):
        df = quandl.get("CHRIS/CME_CL8")
        self.assertGreater(df.index[-1], datetime.datetime.strptime('2020-01-01', '%Y-%m-%d'))

    def test_get_data(self):
        df = qf.get_data({'CHRIS/CME_' + k: k for k in ['CL1', 'CL2', 'CL3', 'CL4', 'CL5']}, 'Last').head()
        self.assert_frame_equal('get_data_df.csv', df)


if __name__ == '__main__':
    unittest.main()
