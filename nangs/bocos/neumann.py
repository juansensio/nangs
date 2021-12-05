import numpy as np
import torch
from .boco import Boco


class Neumann(Boco):
    def __init__(self, sampler, name="neumann"):
        super().__init__(name)
        self.vars = sampler.vars
        self.sampler = sampler

    def sample(self, n_samples=None):
        return self.sampler.sample(n_samples)

    def validate(self, inputs, outputs):
        super().validate()
        assert inputs == self.vars, f'Boco {self.name} with different inputs !'
        assert self.computeBocoLoss, "You need to specify a function to compute the loss"

    def computeLoss(self, model, criterion, inputs, outputs):
        _X = self.sample()
        X = torch.stack([
            _X[var]
            for var in inputs
        ], axis=-1)
        X.requires_grad_(True)
        y = model(X)
        loss = self.computeBocoLoss(X, y)
        return {f'{self.name}_{name}': criterion(l, torch.zeros(l.shape).to(X.device)) for name, l in loss.items()}

    def computeGrads(self, outputs, inputs):
        grads, = torch.autograd.grad(outputs, inputs,
                                     grad_outputs=outputs.data.new(
                                         outputs.shape).fill_(1),
                                     create_graph=True, only_inputs=True)
        return grads
