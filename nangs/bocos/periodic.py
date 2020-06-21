import numpy as np
import torch
from .boco import Boco


class PeriodicDataset(torch.utils.data.Dataset):
    def __init__(self, data, device="cpu"):
        X1 = np.stack(np.meshgrid(*data[0]), -1).reshape(-1, len(data[0]))
        X2 = np.stack(np.meshgrid(*data[1]), -1).reshape(-1, len(data[1]))
        assert len(X1) == len(
            X2), "the length of the inputs don't match"
        self.X1 = torch.from_numpy(X1).float().to(device)
        self.X2 = torch.from_numpy(X2).float().to(device)

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, ix):
        return self.X1[ix], self.X2[ix]


class Periodic(Boco):
    def __init__(self, x1, x2, device="cpu", name="periodic"):
        super().__init__(name)
        assert isinstance(x1, dict), "you must pass a dict with your data"
        assert isinstance(x2, dict), "you must pass a dict with your data"
        assert x1.keys() == x2.keys(), "both inputs must have the same variables"
        self.vars = [tuple(x1.keys()), tuple(x2.keys())]
        data = [x1.values(), x2.values()]
        self.dataset = PeriodicDataset(data, device)

    def validate(self, inputs, outputs):
        super().validate()
        _inputs1, _inputs2 = self.vars
        assert inputs == _inputs1, f'Boco {self.name} with different inputs !'
        assert inputs == _inputs2, f'Boco {self.name} with different inputs !'

    def computeLoss(self, batch, model, criterion):
        x1, x2 = batch
        p1 = model(x1)
        p2 = model(x2)
        return {self.name: criterion(p1, p2)}
