import logging
import math
from typing import Sequence, Iterable, Tuple, Callable
import numpy as np
import torch

from utils import lazy_property

logger = logging.getLogger(__file__)


def as_array(data) -> np.ndarray:
    return data if isinstance(data, np.ndarray) else np.asarray(data)


class TrainingData:

    def __init__(self, X: Iterable[Sequence[float]], Y: Iterable[float], pct_test: float = 0.2,
                 pct_validation: float = 0.1, device: str = None) -> None:
        """
        :param X: Independent variables
        :param Y: Predicted Y values.
        :param pct_test: Portion for test set
        :param pct_validation: Portion for validation set
        """
        super().__init__()
        # todo in TF book test and validation are opposite sets
        self.pct_test = pct_test
        self.pct_validation = pct_validation
        self.X = as_array(X)
        self.Y = as_array(Y)
        n, self.n_features = self.X.shape
        self.n = n
        self.n_train = int(np.round(n * (1 - pct_test - pct_validation)))
        self.n_test = int(np.round(n * pct_test))
        self.n_validation = self.n - self.n_train - self.n_test
        self.X_train, self.X_test, self.X_validation = np.vsplit(self.X, [self.n_train, self.n_train + self.n_test])
        self.Y_train, self.Y_test, self.Y_validation = np.split(self.Y, [self.n_train, self.n_train + self.n_test])
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f'GPU detected. Running on {device}')
            else:
                device = 'cpu'
                logger.info('No GPU detected. Running on CPU')
        self.device = device  # todo remove this field

    def _X_tensor(self, X) -> torch.Tensor:
        # import torch
        # # Otherwise default dtype would be torch.float32 which influence a lot MSELoss function value
        # torch.set_default_dtype(torch.float64)
        # Also, we need to change torch.from_numpy(...).float() -> torch.from_numpy(...).double()
        return torch.from_numpy(X).float()

    def _Y_tensor(self, Y) -> torch.Tensor:
        return torch.from_numpy(Y).float().unsqueeze(dim=1)

    @lazy_property
    def X_train_tensor(self) -> torch.Tensor:
        return self._X_tensor(self.X_train)

    @lazy_property
    def Y_train_tensor(self) -> torch.Tensor:
        return self._Y_tensor(self.Y_train)

    @lazy_property
    def X_test_tensor(self) -> torch.Tensor:
        """
        Suppose to use test tensor just on a target device (not a cpu)
        :return:
        """
        return self._X_tensor(self.X_test).to(self.device)

    @lazy_property
    def Y_test_tensor(self) -> torch.Tensor:
        """
        Suppose to use test tensor just on a target device (not a cpu)
        :return:
        """
        return self._Y_tensor(self.Y_test).to(self.device)

    @lazy_property
    def Y_validation_tensor(self) -> torch.Tensor:
        return self._Y_tensor(self.Y_validation).to(self.device)

    @lazy_property
    def X_validation_tensor(self) -> torch.Tensor:
        """
        Suppose to use test tensor just on a target device (not a cpu)
        :return:
        """
        return self._X_tensor(self.X_validation).to(self.device)

    def train_batches(self, batch_size: int = None, logging_func: Callable = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is a patch generator to torch.utils.data import DataLoader which is VERY slow on 1st iteration.
        todo Returns data sets which are on device already!
        """
        x = self.X_train_tensor
        y = self.Y_train_tensor
        if batch_size is None:
            yield x.to(self.device), y.to(self.device)
            if logging_func:
                msg = logging_func(1, 1)
                print(f'\r{msg}', end='', flush=True)
        else:
            n_batches = math.ceil(self.n / batch_size)
            offset = 0
            for batch_ndx in range(n_batches):
                offset += batch_size
                yield x[offset:offset + batch_size, :].to(self.device), y[offset:offset + batch_size].to(self.device)
                if logging_func:
                    msg = logging_func(batch_ndx, n_batches)
                    print(f'\r{msg}', end='', flush=True)