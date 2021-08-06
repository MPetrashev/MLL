import torch
from torch.nn import functional as F


class Net(torch.nn.Module):
    def __init__(self, n_features: int, n_hiddens:int, n_layers:int, n_outputs: int, output_layer_bias_shift:int= None):
        """
        :param n_features:
        :param n_hiddens:
        :param n_layers:
        :param n_outputs:
        :param output_layer_bias_shift: Controls initial bias for output layer. User can set it to the mean of training
        PVs to speed up the convergence.
        """
        super(Net, self).__init__()

        self.n_hiddens = n_hiddens
        self.n_layers = n_layers

        self.linears = torch.nn.ModuleList([torch.nn.Linear(n_features, n_hiddens)])
        self.linears.extend([torch.nn.Linear(n_hiddens, n_hiddens) for i in range(1, n_layers)])
        self.output_layer = torch.nn.Linear(n_hiddens, n_outputs)
        self.linears.append(self.output_layer)

        if output_layer_bias_shift:
            self.output_layer.bias.data = torch.add(self.output_layer.bias.data, output_layer_bias_shift)

    def forward(self, x):
        for lin in self.linears[:-1]:
            # MemoryException cause at lin(x) line or F.relu(x)
            x = lin(x)
            x = F.relu(x)           # Activation function for hidden layer
        x = self.output_layer(x)     # Apply last layer without activation
        return x