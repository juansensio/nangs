import torch
import numpy as np
from .sine import Sine

# Fourier Feature Networks
# code: https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb
# paper: https://arxiv.org/abs/2006.10739

# Fourier feature mapping


class FourierFeatureMapping(torch.nn.Module):
    def __init__(self, size, std=1, mult=1, device="cpu"):
        super().__init__()
        self.B = torch.normal(mean=torch.zeros(
            size), std=torch.full(size, std)).to(device)
        self.B = mult*self.B

    def forward(self, x):
        x_proj = (2.*np.pi*x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=1)


def block(i, o, a=None):
    return torch.nn.Sequential(
        torch.nn.Linear(i, o),
        Sine() if a == "sine" else torch.nn.ReLU()
    )


class FFN(torch.nn.Module):
    def __init__(self, inputs, outputs, layers, neurons, activation=None, ffm_size=None, std=1, mult=1, device="cpu"):
        super().__init__()
        if ffm_size == None:
            ffm_size = neurons // 2
        ffm = FourierFeatureMapping((ffm_size, inputs), std, mult, device)
        fc_hidden = [
            block(neurons, neurons, activation)
            for layer in range(layers-1)
        ]
        fc_out = block(neurons, outputs)

        self.ffn = torch.nn.Sequential(
            ffm,
            *fc_hidden,
            fc_out
        )

    def forward(self, x):
        return self.ffn(x)
