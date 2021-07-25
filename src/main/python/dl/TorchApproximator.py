import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gc

from utils import as_ndarray, vrange

logger = logging.getLogger(__file__)


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_layers, n_output):
        super(Net, self).__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.linears = torch.nn.ModuleList([torch.nn.Linear(n_feature, n_hidden)])
        self.linears.extend([torch.nn.Linear(n_hidden, n_hidden) for i in range(1, n_layers)])
        self.linears.append(torch.nn.Linear(n_hidden, n_output))

    def forward(self, x):
        for lin in self.linears[:-1]:
            # MemoryException cause at lin(x) line or F.relu(x)
            x = lin(x)
            x = F.relu(x)              # Activation function for hidden layer
        x = self.linears[-1](x)             # Apply last layer without activation
        return x


def wipe_memory(optimizer):
    """
    Example taken from here: https://discuss.pytorch.org/t/how-can-we-release-gpu-memory-cache/14530/27
    todo: but didn't use
        def _optimizer_to(self, device)
    as soon as it works without this method yet. But needs to wait some time. May be 30 secs
    :return:
    """
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f'GPU cache is cleared')


def run_batch(net, x_, y_, x_test_, y_test_, loss_func, optimizer):
    prediction = net(x_)
    loss = loss_func(prediction, y_)

    prediction_test = net(x_test_)
    loss_test = loss_func(prediction_test, y_test_).data.cpu().numpy().item()

    optimizer.zero_grad()
    loss.backward() # MemoryException can arrise here as well
    optimizer.step()
    return loss.data.cpu().numpy().item(), loss_test


# todo move device to object var
def fit_net(net: Net, n_epochs: int, x: torch.tensor, y: torch.tensor, pct_test: float, pct_validation: float, lr=0.01
            , batch_size: int = None, shuffle=False, device: str='cpu'):

    n = y.size()[0]
    n_train = int(np.round(n * (1 - pct_test - pct_validation))) # todo rename
    if batch_size is None:
        batch_size = n_train # run non-Stochastic GD
    n_test = int(np.round(n * pct_test))

    x_train = x[:n_train]
    x_test = x[n_train:(n_train + n_test)]
    y_train = y[:n_train]
    y_test = y[n_train:(n_train + n_test)]

    net.to(device) # todo do we need it?

    x_test_ = x_test.to(device)
    y_test_ = y_test.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()  # Loss/cost function
    try:

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
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            losses.append([epoch + 1, loss_test, loss])

        # pass whole train set to GPU if it's NOT Stochastic GD
        if batch_size == n_train:
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            for epoch in vrange(n_epochs, extra_info=lambda idx: f'Best loss test: {best_loss_test}'):
                loss, loss_test = run_batch(net, x_train, y_train, x_test_, y_test_, loss_func, optimizer)
                record_step(epoch, loss, loss_test)
        else:
            if n_train % batch_size != 0:
                raise ValueError(f"Sorry, we don't support non-integer number of batches at this moment: "
                                 f"{n_train}(Number of samples) % {batch_size}(Batch size) should be zero")
            for epoch in vrange(n_epochs, extra_info=lambda idx: f'Best loss test: {best_loss_test}'):
                for batch_ndx in range(int(n_train / batch_size)):
                    offset = batch_ndx * batch_size
                    x_ = x_train[offset:offset+batch_size, :].to(device)
                    y_ = y_train[offset:offset+batch_size].to(device)
                    loss, loss_test = run_batch(net, x_, y_, x_test_, y_test_, loss_func, optimizer)
                    record_step(epoch, loss, loss_test)
    finally:
        wipe_memory(optimizer)
    return best_loss_test, checkpoint, pd.DataFrame(losses, columns=['Epoch', 'Loss Test', 'Loss'])


class TorchApproximator:

    def __init__(self, seed: int = 314, device: str = None) -> None:
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
                logger.info(f'GPU detected. Running on {device}')
            else:
                device = 'cpu'
                logger.info('No GPU detected. Running on CPU')
        self.device = device

        self.seed = seed

    def train(self, states, pvs, n_epochs=6000, pct_test=0.2  # Portion for test set
              , pct_validation=0.1  # Portion for validation set
              , n_hidden: int = 100, n_layers: int = 4, lr=0.01, batch_size: int = None):
        self.samples_t = torch.from_numpy(states).float() #todo rename samples and values to x and y
        self.values_t = torch.from_numpy(pvs).float().unsqueeze(dim=1)
        self.pvs = pvs
        np.random.seed(self.seed)
        self.generator = torch.manual_seed(self.seed)

        self.pct_validation = pct_validation
        n_features = states.shape[1]
        net = Net(n_feature=n_features, n_hidden=n_hidden, n_layers=n_layers, n_output=1)  # define the network
        best_loss_test, checkpoint, df = fit_net(net, n_epochs, self.samples_t, self.values_t, pct_test, pct_validation, lr=lr
                                     , device=self.device, batch_size=batch_size)
        checkpoint['n_features'] = n_features
        return checkpoint, df, best_loss_test

    def load_model(self, checkpoint):
        model = Net(n_feature=checkpoint['n_features'],
                    n_hidden=checkpoint['n_hidden'],
                    n_layers=checkpoint['n_layers'],
                    n_output=1)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(self.device)
        return model

    def validation_set(self, model):
        n = self.values_t.size()[0]
        ind_validation = int(np.round(n * (1 - self.pct_validation)))
        samples_validation = self.samples_t[ind_validation:].to(self.device)
        approximation = model(samples_validation).flatten().data.cpu().numpy()  # approximated PVs
        return self.pvs[ind_validation:], approximation
