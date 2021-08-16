import copy
import logging
import sys
from collections import namedtuple
from typing import Callable, Tuple, Dict

import torch
import numpy as np
import pandas as pd

from dl.Net import Net
from dl.TrainingData import TrainingData
from dl.cuda import optimizer_to

logger = logging.getLogger(__file__)
# todo change data.cpu().numpy().item() on item()
Training_Info = namedtuple('Training_Info', ['loss_test', 'epoch_idx', 'last_epoch'])


class TorchApproximator:

    def __init__(self, data: TrainingData, n_layers: int, n_hiddens: int, seed: int = 314, batch_size: int = None,
                 loss_func: Callable = None, stop_condition: Callable = None, **kwargs) -> None:
        super().__init__()
        self.device = data.device
        self._data = data
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        self.n_features = data.n_features
        self.seed = seed
        self.batch_size = batch_size
        self.loss_func = loss_func if loss_func else torch.nn.MSELoss()  # Loss/cost function
        self.stop_condition = stop_condition
        self.net = Net(self.n_features, self.n_layers, self.n_hiddens, 1)
        self.__dict__.update(kwargs)  # todo do we need it?

    def train(self, n_epochs: int = 6000) -> Tuple[Training_Info, pd.DataFrame]:
        # todo self.values_t would contain [[pv_0], [pv_1],...] instead if [pv_0, pv_1,...] Simplify it
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        return self._fit_net(n_epochs)

    def data(self, device: str = None) -> TrainingData:
        """
        To avoid GPU memory leaking we need to use copy of data or put back all field values to cpu
        :return:
        """
        result = copy.copy(self._data)
        if device:
            result.device = device
        return result

    def calc_loss(self, X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = self.net(X)
        return self.loss_func(prediction, Y), prediction

    # Example taken from here: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
    def _run_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor, data: TrainingData) \
            -> Tuple[float, float, bool]:
        loss, _ = self.calc_loss(x_batch, y_batch)
        loss_test, prediction_test = self.calc_loss(data.X_test_tensor, data.Y_test_tensor)

        self.optimizer.zero_grad()
        loss.backward()  # MemoryException can arise here as well
        self.optimizer.step()
        return loss.item(), loss_test.item(), self.stop_condition and self.stop_condition(loss_test, data.Y_test, prediction_test)

    # todo move device to object var
    def _fit_net(self, n_epochs: int) -> Tuple[Training_Info, pd.DataFrame]:
        """
        :param n_epochs:
        :return: Tuple of: best_epoch_info and loss history pd.DataFrame
        """
        # todo check SGD+Nesterov optimizer About SGD+Nesterov: https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc
        # Correct value of momentum is obtained by cross validation and would avoid getting stuck in a local minima.
        # In fact it is said that SGD+Nesterov can be as good as Adam’s technique.
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)

        best_epoch = {
            'loss_test': sys.float_info.max
        }
        losses_history = []
        epoch = -1

        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.01)
        try:
            self.net.to(self.device)  # todo do we need it?
            data = self.data()

            stop = False
            for epoch in range(n_epochs):
                for x_batch, y_batch in data.train_batches(self.batch_size, lambda batch_ndx=None,
                        n_batches=None: f'{(epoch*n_batches+batch_ndx)/(n_epochs*n_batches)*100.:5.2f}% completed. Best loss test: {best_epoch["loss_test"]:.7f}'):
                    loss_train, loss_test, stop = self._run_batch(x_batch, y_batch, data)
                    if loss_test < best_epoch['loss_test']:
                        best_epoch['loss_test'] = loss_test
                        best_epoch['epoch_idx'] = epoch
                    losses_history.append([epoch + 1, loss_test, loss_train])
                    if stop:
                        logger.info(f'Stop condition achieved at {loss_test} lost_test value')
                        break
                if stop:  # todo not ideal double stop from inner for but there is no better way to do it
                    break
        finally:
            optimizer_to(self.optimizer, torch.device('cpu'))
            del self.optimizer  # todo if we did create it ourself, it's fine. Otherwise? Also, we can't re-train
            best_epoch['last_epoch'] = epoch
            # does fix memory leaking problem: gpu_memory_info()['Allocated'] wouldn't 0
            # net.load_state_dict = {k: v.to('cpu') for k, v in net.state_dict().items()}
            self.net.to('cpu')
            # wipe_memory()
        return Training_Info(**best_epoch), pd.DataFrame(losses_history, columns=['Epoch', 'Loss Test', 'Loss'])
