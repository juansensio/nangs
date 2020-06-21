import numpy as np
import torch
from .boco import Boco


class NeumannDataset(torch.utils.data.Dataset):
    def __init__(self, data, device="cpu"):
        X = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))
        self.X = torch.from_numpy(X).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix]


class Neumann(Boco):
    def __init__(self, x, device="cpu", name="neumann"):
        super().__init__(name)
        assert isinstance(x, dict), "you must pass a dict with your data"
        self.vars = tuple(x.keys())
        data = x.values()
        self.dataset = NeumannDataset(data, device)
        self.device = device

    def validate(self, inputs, outputs):
        super().validate()
        assert inputs == self.vars, f'Boco {self.name} with different inputs !'
        assert self.computeBocoLoss, "You need to specify a function to compute the loss"

    def computeLoss(self, batch, model, criterion):
        inputs = batch
        inputs.requires_grad = True
        outputs = model(inputs)
        loss = self.computeBocoLoss(inputs, outputs)
        return {f'{self.name}_{name}': criterion(l, torch.zeros(l.shape).to(self.device)) for name, l in loss.items()}

    def computeGrads(self, outputs, inputs):
        grads, = torch.autograd.grad(outputs, inputs,
                                     grad_outputs=outputs.data.new(
                                         outputs.shape).fill_(1),
                                     create_graph=True, only_inputs=True)
        return grads
