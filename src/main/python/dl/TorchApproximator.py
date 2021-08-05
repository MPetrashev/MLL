import logging
import math
import sys
from typing import Callable

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc

from utils import vrange

logger = logging.getLogger(__file__)
# todo change data.cpu().numpy().item() on item()


def DataLoader(x, y, batch_size=None):
    """
    This is a patch generator to torch.utils.data import DataLoader which is VERY slow on 1st iteration
    """
    if batch_size is None:
        yield x, y
    else:
        n_batches = math.ceil(len(y) / batch_size)
        for batch_ndx in range(n_batches):
            offset = batch_ndx * batch_size
            yield x[offset:offset+batch_size, :], y[offset:offset+batch_size]


class Net(torch.nn.Module):
    def __init__(self, n_features: int, n_hiddens:int, n_layers:int, n_outputs: int, output_layer_bias_shift: int = None):
        """
        :param n_features: 
        :param n_hiddens: 
        :param n_layers: 
        :param n_outputs: 
        :param output_layer_bias_shift: Controls initial bias for output layer. User can set it to the mean of training
        PVs to speed up the convergence.
        """
        super(Net, self).__init__()

        self.n_hidden = n_hiddens
        self.n_layers = n_layers

        self.linears = torch.nn.ModuleList([torch.nn.Linear(n_features, n_hiddens)])
        self.linears.extend([torch.nn.Linear(n_hiddens, n_hiddens) for i in range(1, n_layers)])
        self.linears.append(torch.nn.Linear(n_hiddens, n_outputs))

        if output_layer_bias_shift:
            for layer in self.linears[-1:]:
                layer.bias.data = torch.add(layer.bias.data, output_layer_bias_shift)

    def forward(self, x):
        for lin in self.linears[:-1]:
            # MemoryException cause at lin(x) line or F.relu(x)
            x = lin(x)
            x = F.relu(x)              # Activation function for hidden layer
        x = self.linears[-1](x)             # Apply last layer without activation
        return x


# Example taken from here: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
def optimizer_to(optimizer, device):
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def gpu_memory_info():
    """
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    :return:
    """
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.memory_reserved(0)
    allocated = torch.cuda.memory_allocated(0)
    return {
        'Free': reserved - allocated,
        'Total': total,
        'Reserved': reserved,
        'Allocated': allocated
    }


def wipe_memory():
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f'GPU cache is cleared')


def run_batch(net, x_, y_, x_test, y_test, loss_func, optimizer, test_batch_size=None, device='cpu'):
    prediction = net(x_)
    loss = loss_func(prediction, y_)

    loss_test = sys.float_info.max
    for x_test_batch, y_test_batch in DataLoader(x_test, y_test, batch_size=test_batch_size):
        prediction_test = net(x_test_batch.to(device))
        loss_test_ = loss_func(prediction_test, y_test_batch.to(device)).data.cpu().numpy().item()
        if loss_test_ < loss_test:
            loss_test = loss_test_

    optimizer.zero_grad()
    loss.backward()  # MemoryException can arise here as well
    optimizer.step()
    return loss.data.cpu().numpy().item(), loss_test


# todo move device to object var
def fit_net(net: Net, n_epochs: int, x: torch.tensor, y: torch.tensor, pct_test: float, pct_validation: float,
            loss_func : Callable, lr=0.01, batch_size: int = None, shuffle=False, device: str='cpu'):

    n = len( y ) #.size()[0]
    n_train = int(np.round(n * (1 - pct_test - pct_validation))) # todo rename
    n_test = int(np.round(n * pct_test))


    x_train = x[:n_train]
    x_test = x[n_train:(n_train + n_test)]
    y_train = y[:n_train]
    y_test = y[n_train:(n_train + n_test)]



    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    try:
        net.to(device)  # todo do we need it?

        best_loss_test = y.abs().max().item()
        checkpoint = {}
        losses = []

        def record_step(epoch, loss, loss_test):
            nonlocal best_loss_test, checkpoint
            if loss_test < best_loss_test:
                best_loss_test = loss_test
                checkpoint = {
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'model_state_dict': net.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                }
            losses.append([epoch + 1, loss_test, loss])

        for epoch in vrange(n_epochs, extra_info=lambda idx: f'Best loss test: {best_loss_test}'):
            for x_batch, y_batch in DataLoader(x_train, y_train, batch_size=batch_size):
                loss, loss_test = run_batch(net, x_batch.to(device), y_batch.to(device), x_test, y_test, loss_func
                            , optimizer, device=device
                            , test_batch_size=math.ceil( batch_size * n_test / n_train ) if batch_size else None)
                record_step(epoch, loss, loss_test)
    finally:
        optimizer_to(optimizer, torch.device('cpu'))
        del optimizer
        net.to('cpu')
        if 'model_state_dict' in checkpoint:
            checkpoint['model_state_dict'] = { k : v.to('cpu') for k, v in checkpoint['model_state_dict'].items()} # does fix memory leaking problem: gpu_memory_info()['Allocated'] wouldn't 0
        # wipe_memory()

    return best_loss_test, checkpoint, pd.DataFrame(losses, columns=['Epoch', 'Loss Test', 'Loss'])


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
              , n_hiddens: int = 100, n_layers: int = 4, lr=0.01, batch_size: int = None):
        self.samples_t = torch.from_numpy(states).float() #todo rename samples and values to x and y
        self.values_t = torch.from_numpy(pvs).float().unsqueeze(dim=1)
        self.pvs = pvs
        np.random.seed(self.seed)
        self.generator = torch.manual_seed(self.seed)

        self.pct_validation = pct_validation
        n_features = states.shape[1]
        net = Net(n_features=n_features, n_hiddens=n_hiddens, n_layers=n_layers, n_outputs=1)  # define the network
        best_loss_test, checkpoint, df = fit_net(net, n_epochs, self.samples_t, self.values_t, pct_test, pct_validation,
                                    self.loss_func, lr=lr, device=self.device, batch_size=batch_size)
        checkpoint['n_features'] = n_features
        return checkpoint, df, best_loss_test

    def load_model(self, checkpoint):
        model = Net(n_features=checkpoint['n_features'],
                    n_hiddens=checkpoint['n_hidden'],
                    n_layers=checkpoint['n_layers'],
                    n_outputs=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        return model

    def validation_set(self, model):
        n = len(self.values_t)
        ind_validation = int(np.round(n * (1 - self.pct_validation)))
        samples_validation = self.samples_t[ind_validation:].to(self.device)
        approximation = model(samples_validation).flatten().data.cpu().numpy()  # approximated PVs
        return self.pvs[ind_validation:], approximation
