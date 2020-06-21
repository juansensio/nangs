import numpy as np
import torch
from .boco import Boco


class DirichletDataset(torch.utils.data.Dataset):
    def __init__(self, data, device="cpu"):
        X = np.stack(np.meshgrid(*data[0]), -1).reshape(-1, len(data[0]))
        Y = np.stack(data[1], axis=1)
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
        self.vars = [tuple(x.keys()), tuple(y.keys())]
        data = [x.values(), list(y.values())]
        self.dataset = DirichletDataset(data, device)

    def validate(self, inputs, outputs):
        super().validate()
        _inputs, _outputs = self.vars
        assert inputs == _inputs, f'Boco {self.name} with different inputs !'
        assert outputs == _outputs, f'Boco {self.name} with different outputs !'

    def computeLoss(self, batch, model, criterion):
        x, y = batch
        p = model(x)
        return {self.name: criterion(p, y)}
