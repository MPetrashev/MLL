import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

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
            x = F.relu(lin(x))       # Activation function for hidden layer
        x = self.linears[-1](x)             # Apply last layer without activation
        return x


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

    def fit_net(self, net: Net, n_epochs: int, x: torch.tensor, y: torch.tensor, pct_test: float,
                pct_validation: float):

        n = y.size()[0]
        n_train = int(np.round(n * (1 - pct_test - pct_validation)))
        n_test = int(np.round(n * pct_test))

        x_train = x[:n_train]
        x_test = x[n_train:(n_train + n_test)]
        y_train = y[:n_train]
        y_test = y[n_train:(n_train + n_test)]

        net.to(self.device)
        x_ = x_train.to(self.device)
        y_ = y_train.to(self.device)

        x_test_ = x_test.to(self.device)
        y_test_ = y_test.to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        loss_func = torch.nn.MSELoss()

        best_l = y.abs().max().item()
        checkpoint = {}
        losses = []

        for e in range(n_epochs):
            prediction = net(x_)
            loss = loss_func(prediction, y_)

            prediction_test = net(x_test_)
            loss_test = loss_func(prediction_test, y_test_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l = loss_test.data.cpu().numpy()
            if l.item() < best_l:
                best_l = l.item()
                checkpoint = {
                    'n_hidden': net.n_hidden,
                    'n_layers': net.n_layers,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
            losses.append([e + 1, l.item(), loss.data.cpu().numpy().item()])

        return best_l, checkpoint, pd.DataFrame(losses, columns=['Epoch', 'Loss Test', 'Loss'])

    def train(self, states, pvs, n_epochs=6000, pct_test=0.2  # Portion for test set
              , pct_validation=0.1  # Portion for validation set
              , n_hidden: int = 1500, n_layers: int = 4):
        self.pvs = pvs
        np.random.seed(self.seed)
        self.generator = torch.manual_seed(self.seed)

        self.pct_validation = pct_validation
        self.samples_t = torch.from_numpy(states.T).float()
        self.values_t = torch.from_numpy(pvs).float().unsqueeze(dim=1)
        n_features = states.shape[0]
        net = Net(n_feature=n_features, n_hidden=n_hidden, n_layers=n_layers, n_output=1)  # define the network
        ls, checkpoint, df = self.fit_net(net, n_epochs, self.samples_t, self.values_t, pct_test, pct_validation)
        checkpoint['n_features'] = n_features
        return checkpoint, df

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
