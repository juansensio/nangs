import torch
import numpy as np
from .sine import Sine


def block(i, o):
    fc = torch.nn.Linear(i, o)
    torch.nn.init.uniform_(fc.weight, -np.sqrt(6 / i), np.sqrt(6 / i))
    return torch.nn.Sequential(
        Sine(),
        fc
    )


class Siren(torch.nn.Module):
    def __init__(self, inputs, outputs, layers, neurons, omega_0=30.):
        super().__init__()
        fc_in = torch.nn.Linear(inputs, neurons)
        torch.nn.init.uniform_(fc_in.weight, -omega_0 /
                               inputs, omega_0 / inputs)
        fc_hidden = [
            block(neurons, neurons)
            for layer in range(layers-1)
        ]
        fc_out = block(neurons, outputs)

        self.mlp = torch.nn.Sequential(
            fc_in,
            *fc_hidden,
            fc_out
        )

    def forward(self, x):
        return self.mlp(x)

# SIREN
# code: https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=zwpojK1wAJg1
# paper: https://arxiv.org/abs/2006.09661


class SineLayer(torch.nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class _Siren(torch.nn.Module):
    def __init__(self, inputs, outputs, layers, neurons, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(inputs, neurons,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(layers-1):
            self.net.append(SineLayer(neurons, neurons,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = torch.nn.Linear(neurons, outputs)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / neurons) / hidden_omega_0,
                                             np.sqrt(6 / neurons) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(neurons, outputs,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        y = self.net(x)
        return y
