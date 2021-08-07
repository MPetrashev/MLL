import logging
import math
import sys
from collections import namedtuple
from typing import Callable, Dict

import torch
import numpy as np
import pandas as pd

from dl.DataLoader import DataLoader
from dl.Net import Net
from dl.cuda import optimizer_to
from utils import vrange, less_than_1pc_exceeds_1pc_diff

logger = logging.getLogger(__file__)
# todo change data.cpu().numpy().item() on item()


Checkpoint = namedtuple('Checkpoint', ['n_features', 'n_layers', 'n_hiddens', 'n_outputs', 'model_state_dict']
                        , defaults=(None, None, None, 1, {}))


# Example taken from here: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27


def run_batch(net: Net, x_, y_, x_test, y_test, loss_func, optimizer, test_batch_size=None, device='cpu'
              , stop_condition: Callable = None):
    prediction = net(x_)
    loss = loss_func(prediction, y_)

    if test_batch_size:
        raise ValueError(
            'loss_test accumulates incorrectly. Must be reimplemented: \sum_1^100 (x-x\')^2 break on N subsets')

    loss_test = sys.float_info.max
    for x_test_batch, y_test_batch in DataLoader(x_test, y_test, batch_size=test_batch_size):
        prediction_test = net(x_test_batch.to(device))
        loss_test_ = loss_func(prediction_test, y_test_batch.to(device)).data.cpu().numpy().item()
        if loss_test_ < loss_test:
            loss_test = loss_test_

    optimizer.zero_grad()
    loss.backward()  # MemoryException can arise here as well
    optimizer.step()
    return loss.data.cpu().numpy().item(), loss_test, stop_condition and stop_condition(y_test, prediction_test)


# todo move device to object var
def fit_net(net: Net, n_epochs: int, x: torch.tensor, y: torch.tensor, pct_test: float, pct_validation: float,
            loss_func : Callable, lr=0.01, batch_size: int = None, shuffle=False, device: str='cpu', stop_condition: Callable = None):

    n = y.shape[0]
    n_train = int(np.round(n * (1 - pct_test - pct_validation))) # todo rename
    n_test = int(np.round(n * pct_test))


    x_train = x[:n_train]
    x_test = x[n_train:(n_train + n_test)]
    y_train = y[:n_train]
    y_test = y[n_train:(n_train + n_test)]


    # todo check SGD+Nesterov optimizer About SGD+Nesterov: https://medium.com/@Biboswan98/optim-adam-vs-optim-sgd-lets-dive-in-8dbf1890fbdc
    # Correct value of momentum is obtained by cross validation and would avoid getting stuck in a local minima.
    # In fact it is said that SGD+Nesterov can be as good as Adam’s technique.
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss_test = y.abs().max().item()

    checkpoint = Checkpoint(net.n_features, net.n_layers, net.n_hiddens, 1)
    losses = []
    epoch = -1
    try:
        net.to(device)  # todo do we need it?

        def record_step(epoch, loss, loss_test):
            nonlocal best_loss_test, checkpoint
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                checkpoint = checkpoint._replace(n_layers=net.n_layers, n_hiddens=net.n_hiddens,
                                                 model_state_dict=net.state_dict())
            losses.append([epoch + 1, loss_test, loss])

        stop = False
        for epoch in vrange(n_epochs, extra_info=lambda idx: f'Best loss test: {best_loss_test}'):
            for x_batch, y_batch in DataLoader(x_train, y_train, batch_size=batch_size):
                loss, loss_test, stop = run_batch(net, x_batch.to(device), y_batch.to(device), x_test, y_test, loss_func
                            , optimizer, device=device, stop_condition=stop_condition
                            , test_batch_size=math.ceil( batch_size * n_test / n_train ) if batch_size else None)
                record_step(epoch, loss, loss_test)
                if stop:
                    break
            if stop:    # todo not ideal double stop from inner for but there is no better way to do it
                break
    finally:
        optimizer_to(optimizer, torch.device('cpu'))
        del optimizer
        net.to('cpu')
        if checkpoint.model_state_dict:
            # does fix memory leaking problem: gpu_memory_info()['Allocated'] wouldn't 0
            checkpoint._replace(model_state_dict={k: v.to('cpu') for k, v in checkpoint.model_state_dict.items()})
        # wipe_memory()

    return best_loss_test, checkpoint, epoch, pd.DataFrame(losses, columns=['Epoch', 'Loss Test', 'Loss'])


class TorchApproximator:

    def __init__(self, seed: int = 314, device: str = None, loss_func: Callable = None) -> None:
        super().__init__()
        self.loss_func = loss_func if loss_func else torch.nn.MSELoss()  # Loss/cost function
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f'GPU detected. Running on {device}')
            else:
                device = 'cpu'
                logger.info('No GPU detected. Running on CPU')
        self.device = device

        self.seed = seed

    def train(self, states, pvs, n_epochs=6000, pct_test=0.2  # Portion for test set
              , pct_validation=0.1  # Portion for validation set
              , n_hiddens: int = 100, n_layers: int = 4, lr=0.01, batch_size: int = None
              , stop_condition: Callable = less_than_1pc_exceeds_1pc_diff):
        # todo self.values_t would contain [[pv_0], [pv_1],...] instead if [pv_0, pv_1,...] Simplify it
        self.samples_t = torch.from_numpy(states).float() #todo rename samples and values to x and y
        self.values_t = torch.from_numpy(pvs).float().unsqueeze(dim=1)
        self.pvs = pvs
        np.random.seed(self.seed)
        self.generator = torch.manual_seed(self.seed)

        self.pct_validation = pct_validation
        n_features = states.shape[1]
        net = Net(n_features=n_features, n_hiddens=n_hiddens, n_layers=n_layers, n_outputs=1)  # define the network
        best_loss_test, checkpoint, epoch, df = fit_net(net, n_epochs, self.samples_t, self.values_t, pct_test
                    , pct_validation, self.loss_func, lr=lr, device=self.device, batch_size=batch_size
                    , stop_condition=stop_condition)
        return checkpoint, df, best_loss_test, epoch  # todo sync order of returned vars with fit_net

    def load_model(self, checkpoint: Checkpoint):
        model = Net(**checkpoint._asdict())
        model.eval()
        model.to(self.device)
        return model

    def validation_set(self, model):
        n = self.values_t.shape[0]
        ind_validation = int(np.round(n * (1 - self.pct_validation)))
        samples_validation = self.samples_t[ind_validation:].to(self.device)
        approximation = model(samples_validation).flatten().data.cpu().numpy()  # approximated PVs
        return self.pvs[ind_validation:], approximation