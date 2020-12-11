from TestCase import TestCase

import unittest
import numpy as np
import pandas as pd
import torch

from dl import TorchApproximator
from dl.TorchApproximator import Net
from scipy.stats import norm


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
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    print(f"GPU detected. Running on {device}")
else:
    print("No GPU detected. Running on CPU")
device="cuda:0"

class ApproximatorTest(TestCase):

    def test_put_prices(self):
        np.random.seed(seed)

        n_samples = 100000  # Total number of samples
        domain = {
            "spot": (0.5, 2),
            "time": (0, 3.0),
            "sigma": (0.1, 0.5),
            "rate": (-0.01, 0.03),
            "div": (0, 0.02)
        }
        samples = np.zeros(shape=(len(domain.keys()), n_samples))
        for i, r in enumerate(domain.values()):
            samples[i] = np.random.uniform(r[0], r[1], n_samples)
        values = v_put(K=1, S=samples[0], T=samples[1], vol=samples[2], r=samples[3], q=samples[4])
        df = pd.DataFrame.from_dict({'PV' : values,'S':samples[0], 'T':samples[1], 'vol':samples[2], 'r':samples[3], 'q':samples[4]})
        self.assert_frame_equal('put_prices.csv', df)

    def test_BS_example(self):
        np.random.seed(seed)
        torch.manual_seed(seed)

        approximator = TorchApproximator()
        df = self.get_test_data('put_prices.csv')
        checkpoint, df = approximator.train(df.drop(columns=['PV']).values.T, df.PV.values, n_epochs=6000, n_hidden=100)
        self.assert_frame_equal('bs_example.csv', df)
        
    def test_torch_approximator(self):
        torch.manual_seed(seed)

        approximator = TorchApproximator()
        net = Net(n_feature=5, n_hidden=100, n_layers=4, n_output=1)  # define the network
        n_epochs = 6000
        pct_test = 0.2  # Portion for test set
        pct_validation = 0.1  # Portion for validation set
        df = self.get_test_data('put_prices.csv')
        values = df.PV.values
        samples = df.iloc[:,1:].values.T
        samples_t = torch.from_numpy(samples.T).float()
        values_t = torch.from_numpy(values).float().unsqueeze(dim=1)

        ls, checkpoint, df = approximator.fit_net(net, n_epochs, samples_t, values_t, pct_test, pct_validation, device)
        self.assert_frame_equal('torch_steps.csv', df)


if __name__ == '__main__':
    unittest.main()
