import numpy as np
import torch
from .boco import Boco


class DirichletDataset(torch.utils.data.Dataset):
    def __init__(self, data, device="cpu"):
        X = np.stack(np.meshgrid(*data[0]), -1).reshape(-1, len(data[0]))
        Y = np.stack(np.meshgrid(*data[1]), -1).reshape(-1, len(data[1]))
        assert len(X) == len(
            Y), "the length of the inputs and outputs don't match"
        self.X = torch.from_numpy(X).float().to(device)
        self.Y = torch.from_numpy(Y).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix], self.Y[ix]


class Dirichlet(Boco):
    def __init__(self, x, y, device="cpu", name="dirichlet"):
        super().__init__(name)
        assert isinstance(x, dict), "you must pass a dict with your data"
        assert isinstance(y, dict), "you must pass a dict with your data"
        self.vars = [list(x.keys()), list(y.keys())]
        data = [x.values(), y.values()]
        self.dataset = DirichletDataset(data, device)

    def computeLoss(self, batch, model, criterion):
        x, y = batch
        p = model(x)
        return criterion(p, y)
