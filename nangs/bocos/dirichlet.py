import numpy as np
import torch
from .boco import Boco


class Dirichlet(Boco):
    def __init__(self, sampler, output_fn, name="dirichlet"):
        super().__init__(name)
        self.vars = sampler.vars
        self.sampler = sampler
        assert callable(output_fn), 'output_fn must be callable'
        self.output_fn = output_fn

    def validate(self, inputs, outputs):
        super().validate()
        assert inputs == self.vars, f'Boco {self.name} with different inputs !'
        _outputs = tuple(self.output_fn(self.sampler.sample(1)).keys())
        if outputs != _outputs:
            print(
                f'Boco {self.name} with different outputs ! {outputs} vs {_outputs}')
        # puedo fitear solo algunos outputs
        # self.output_ids = [outputs.index(v)
        #                    for v in self.vars[1] if v in outputs]

    def sample(self, n_samples=None):
        inputs = self.sampler.sample(n_samples)
        outputs = self.output_fn(inputs)
        return inputs, outputs

    def computeLoss(self, model, criterion, inputs, outputs):
        _X, _y = self.sample()
        X = torch.stack([
            _X[var]
            for var in inputs
        ], axis=-1)
        y_hat = model(X)
        __y = []
        for i, var in enumerate(outputs):
            if var in _y:
                __y.append(_y[var])
            else:
                __y.append(y_hat[:, i])
        y = torch.stack(__y, axis=-1)

        return {self.name: criterion(y, y_hat)}
