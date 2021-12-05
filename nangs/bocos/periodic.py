import numpy as np
import torch
from .boco import Boco


class Periodic(Boco):
    def __init__(self, sampler, sampler1, sampler2, name="periodic"):
        super().__init__(name)
        self.sampler = sampler
        self.sampler1 = sampler1
        self.sampler2 = sampler2

        inputs1 = tuple(self.sampler1.sample(1).keys())
        inputs2 = tuple(self.sampler2.sample(1).keys())
        vars1, vars2 = sampler.vars + inputs1, sampler.vars + inputs2
        assert len(vars1) == len(
            vars2), 'Samplers must have the same variables'
        for var in vars1:
            assert var in vars2, 'Samplers must have the same variables'
        self.vars = vars1

    def sample(self, n_samples=None):
        shared = self.sampler.sample(n_samples)
        inputs = self.sampler1.sample(n_samples)
        inputs.update(shared)
        outputs = self.sampler2.sample(n_samples)
        outputs.update(shared)
        return inputs, outputs

    def validate(self, inputs, outputs):
        super().validate()
        assert len(inputs) == len(
            self.vars), f'Boco {self.name} with different inputs !'
        for var in self.vars:
            assert var in inputs, f'Boco {self.name} with different inputs !'

    def computeLoss(self, model, criterion, inputs, outputs):
        _x1, _x2 = self.sample()
        x1 = torch.stack([
            _x1[var]
            for var in inputs
        ], axis=-1)
        x2 = torch.stack([
            _x2[var]
            for var in inputs
        ], axis=-1)
        y1 = model(x1)
        y2 = model(x2)
        return {self.name: criterion(y1, y2)}
