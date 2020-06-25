import torch


class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Sine2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.sin(x).pow(2)
