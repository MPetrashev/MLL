import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

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
    def fit_net(self, net: Net, n_epochs: int, x: torch.tensor, y: torch.tensor,
                pct_test: float, pct_validation: float, device: str = 'cpu'):

        n = y.size()[0]
        n_train = int(np.round(n * (1 - pct_test - pct_validation)))
        n_test = int(np.round(n * pct_test))

        x_train = x[:n_train]
        x_test = x[n_train:(n_train + n_test)]
        y_train = y[:n_train]
        y_test = y[n_train:(n_train + n_test)]

        net.to(device)
        x_ = x_train.to(device)
        y_ = y_train.to(device)

        x_test_ = x_test.to(device)
        y_test_ = y_test.to(device)

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
                    "n_hidden": net.n_hidden,
                    "n_layers": net.n_layers,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
            losses.append([e + 1, l.item()])

        return best_l, checkpoint, pd.DataFrame(losses, columns=['Step', 'Loss'])