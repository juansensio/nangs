from tqdm import tqdm
import torch
from .utils import *
import numpy as np
import matplotlib.pyplot as plt
from .history import History


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class PDE():

    def __init__(self, inputs, outputs):

        # check lists of unique strings, non-repeated
        if isinstance(inputs, str):
            inputs = tuple(inputs)
        if isinstance(outputs, str):
            outputs = tuple(outputs)

        checkIsListOfStr(inputs)
        checkIsListOfStr(outputs)
        checkUnique(inputs)
        checkUnique(outputs)
        checkNoRepeated(inputs, outputs)

        self.inputs = inputs
        self.outputs = outputs
        self.mesh = None
        self.bocos = []

    def set_sampler(self, sampler):
        assert sampler.vars == self.inputs, "your data does not match the PDE inputs"
        self.sampler = sampler

    def add_boco(self, boco):
        assert boco.name not in [
            boco.name for boco in self.bocos], f'Boco {boco.name} already exists, use another name'
        boco.validate(self.inputs, self.outputs)
        self.bocos.append(boco)

    def update_boco(self, boco):
        for b in self.bocos:
            if b.name == boco.name:
                self.bocos[self.bocos.index(b)] = boco
                return

    def compile(self, model, optimizer, scheduler=None, loss_fn=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn else torch.nn.MSELoss()
        self.scheduler = scheduler

    def computePDELoss(self, vars, grads):
        print("This function need to be overloaded !!!")

    def solve(self, N_STEPS=50):
        # solve PDE
        history = History()
        pbar = tqdm(range(1, N_STEPS + 1))
        for step in pbar:
            history.add({'lr': get_lr(self.optimizer)})
            self.optimizer.zero_grad()
            loss = 0
            # optimize for PDE
            X = self.sampler._sample()
            X.requires_grad_(True)
            y = self.model(X)
            pde_losses = self.computePDELoss(X, y)
            if step == 1:
                assert isinstance(
                    pde_losses, dict), "you should return a dict with the name of the equation and the corresponding loss"
            for name, l in pde_losses.items():
                _loss = self.loss_fn(l, torch.zeros(
                    l.shape).to(self.sampler.device))
                loss += _loss
                history.add_step({name: _loss.item()})
            # optimize for boundary points
            for boco in self.bocos:
                boco_losses = boco.computeLoss(
                    self.model, self.loss_fn, self.inputs, self.outputs)
                for name, l in boco_losses.items():
                    if step == 1:
                        assert isinstance(
                            boco_losses, dict), "you should return a dict with the name of the equation and the corresponding loss"
                    loss += l
                    history.add_step({name: l.item()})
            loss.backward()
            self.optimizer.step()
            pbar.set_description(str(history.average()))
            history.step()
            if self.scheduler:
                self.scheduler.step()
        return history.history

    def computeGrads(self, outputs, inputs):
        grads, = torch.autograd.grad(outputs, inputs,
                                     grad_outputs=outputs.data.new(
                                         outputs.shape).fill_(1),
                                     create_graph=True, only_inputs=True)
        return grads

    def eval(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(X)
