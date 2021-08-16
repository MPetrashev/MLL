from TestCase import TestCase

import unittest
import numpy as np
import pandas as pd
import torch
print( torch.cuda.is_available() ) # todo without this call gpu_memory_info() halts the python process

from dl import TorchApproximator, TFApproximator, shift_pvs, TrainingData
from scipy.stats import norm

from dl.cuda import gpu_memory_info
from utils import lazy_property


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

    @lazy_property
    def put_prices_100K(self):
        return self.get_test_data('put_prices.csv', nrows=100000)

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
        self.assert_frame_equal(self.put_prices_100K, df)

    def test_torch_put_BS_example(self):
        # Original time to execute: 2m 29s
        np.random.seed(seed)
        torch.manual_seed(seed)

        df = self.put_prices_100K

        approximator = TorchApproximator(TrainingData(df.iloc[:, df.columns != 'PV'].values, df.PV.values), 4, 100)
        best_epoch, history = approximator.train(n_epochs=n_epochs)
        self.assert_frame_equal('torch_steps.csv', history, compare_just_head=True)

        data = approximator.data('cpu')
        loss, _ = approximator.calc_loss(data.X_validation_tensor, data.Y_validation_tensor)
        self.assertEqual(loss.item(), 0.008219024166464806)

        # Check shifted PVs approximation
        approximator = TorchApproximator(TrainingData(df.iloc[:, df.columns != 'PV'].values, shift_pvs(df.PV.values)), 4, 100)
        approximator.train(n_epochs)

        data = approximator.data('cpu')
        loss, _ = approximator.calc_loss(data.X_validation_tensor, data.Y_validation_tensor)
        self.assertEqual(loss.item(), 0.12789303064346313)  # todo why shifted is much worse?

    def test_torch_scaled_vs_unscaled(self):
        """
        Best unscaled: 100K, 2, 400. Test: 0.002541386755, Val: 0.002535967855
        Best scaled: 100K,
        :return:
        """

    def test_torch_GPU_memory_leaking(self):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.assertEqual(0, gpu_memory_info()['Allocated'])
        df = self.put_prices_100K
        df = pd.concat([df] * 10, ignore_index=True)

        # check GD
        approximator = TorchApproximator(TrainingData(df.iloc[:, df.columns != 'PV'].values, df.PV.values), 4, 100)
        best_epoch, history1 = approximator.train(30)
        self.assertEqual(0, gpu_memory_info()['Allocated'])
        self.assertIsNotNone(approximator.net)  # Was a cause of leaking GPU memory because of 'model_state_dict' value.

        # check SGD
        approximator = TorchApproximator(TrainingData(df.iloc[:, df.columns != 'PV'].values, df.PV.values), 4, 100
                                         , batch_size=175000)
        best_epoch, history2 = approximator.train(30)
        self.assertEqual(0, gpu_memory_info()['Allocated'])
        self.assertIsNotNone(approximator.net)  # Was a cause of leaking GPU memory because of 'model_state_dict' value.

    def test_tf_put_BS_example(self):
        approximator = TFApproximator()
        df = self.put_prices_100K
        checkpoint, history = approximator.train(df.iloc[:, df.columns != 'PV'], df.PV, n_epochs=n_epochs, n_hidden=100)
        self.assert_frame_equal('tf_steps.csv', history, compare_just_head=True)


if __name__ == '__main__':
    unittest.main()
