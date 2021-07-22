from TestCase import TestCase

import unittest
import numpy as np
import pandas as pd
import torch

from dl import TorchApproximator, TFApproximator
from scipy.stats import norm

from utils import bps


def ds(K, S, T, vol, r, q):
    vol_T = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol * vol) * T) / vol_T
    d2 = d1 - vol_T
    return d1, d2


def put(K, S, T, vol, r, q):
    disc = np.exp(-r * T)
    pv_K = K * disc
    spot_after_div = S * np.exp(-q * T)

    d1, d2 = ds(K, S, T, vol, r, q)
    v = norm.cdf(-d2) * pv_K - norm.cdf(-d1) * spot_after_div
    return v * 100.


v_put = np.vectorize(put)
seed = 314
n_epochs = 6000


class ApproximatorTest(TestCase):

    def test_put_BS_prices(self):
        np.random.seed(seed)

        n_samples = 100000  # Total number of samples
        domain = {
            'spot': (0.5, 2),
            'time': (0, 3.0),
            'sigma': (0.1, 0.5),
            'rate': (-0.01, 0.03),
            'div': (0, 0.02)
        }
        samples = np.zeros(shape=(len(domain.keys()), n_samples))
        for i, r in enumerate(domain.values()):
            samples[i] = np.random.uniform(r[0], r[1], n_samples)
        values = v_put(K=1, S=samples[0], T=samples[1], vol=samples[2], r=samples[3], q=samples[4])
        df = pd.DataFrame.from_dict({'PV' : values,'S':samples[0], 'T':samples[1], 'vol':samples[2], 'r':samples[3], 'q':samples[4]})
        self.assert_frame_equal('put_prices.csv', df)

    def test_torch_put_BS_example(self):
        # Original time to execute: 2m 29s
        np.random.seed(seed)
        torch.manual_seed(seed)

        approximator = TorchApproximator()
        df = self.get_test_data('put_prices.csv')
        checkpoint, history = approximator.train(df.iloc[:, df.columns != 'PV'].values, df.PV.values, n_epochs=n_epochs, n_hidden=100)
        self.assert_frame_equal('torch_steps.csv', history, compare_just_head=True)

        model = approximator.load_model(checkpoint)
        original, approximation = approximator.validation_set(model)
        self.assertLess(max(np.vectorize(bps)(original, approximation)), 80)
        error = approximation-original
        self.assertEqual(np.mean(error), 0.0042583657125902195) # abs((error).mean()) deviation is closed to 0
        self.assertEqual(np.std(error), 0.0905587802924326) # math.sqrt( ((approximation-original)**2).mean() ) mse is small

    def test_tf_put_BS_example(self):
        approximator = TFApproximator()
        df = self.get_test_data('put_prices.csv')
        checkpoint, history = approximator.train(df.iloc[:, df.columns != 'PV'], df.PV, n_epochs=n_epochs, n_hidden=100)
        self.assert_frame_equal('tf_steps.csv', history, compare_just_head=True)


if __name__ == '__main__':
    unittest.main()
