from fastprogress import master_bar, progress_bar
from .history import History
import torch
from .utils import *


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

    def set_mesh(self, mesh):
        assert mesh.vars == self.inputs, "your data does not match the PDE inputs"
        self.mesh = mesh

    def add_boco(self, boco):
        assert boco.name not in [
            boco.name for boco in self.bocos], f'Boco {boco.name} already exists, use another name'
        boco.validate(self.inputs, self.outputs)
        self.bocos.append(boco)

    def compile(self, model, optimizer, scheduler=None, criterion=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion if criterion else torch.nn.MSELoss()
        self.scheduler = scheduler

    def computePDELoss(self, vars, grads):
        print("This function need to be overloaded !!!")

    def solve(self, epochs=50, batch_size=None, shuffle=True):
        dataloaders = self.set_dataloaders(batch_size, shuffle)
        # solve PDE
        history = History()
        mb = master_bar(range(1, epochs+1))
        for epoch in mb:
            history.add({'lr': get_lr(self.optimizer)})
            # iterate over the internal points in batches
            for batch in progress_bar(dataloaders['inner'], parent=mb):
                X = batch
                self.optimizer.zero_grad()
                # optimize for boundary points
                for boco in self.bocos:
                    for batch in dataloaders['bocos'][boco.name]:
                        loss = boco.computeLoss(
                            batch, self.model, self.criterion)
                        for name, l in loss.items():
                            l.backward()
                            history.add_step({name: l.item()})
                # optimize for internal points
                X.requires_grad = True
                p = self.model(X)
                loss = self.computePDELoss(X, p)
                assert isinstance(
                    loss, dict), "you should return a dict with the name of the equation and the corresponding loss"
                for name, l in loss.items():
                    l = self.criterion(l, torch.zeros(
                        l.shape).to(self.mesh.device))
                    l.backward(retain_graph=True)
                    history.add_step({name: l.item()})
                self.optimizer.step()
                mb.child.comment = str(history.average())
            history.step()
            mb.write(f"Epoch {epoch}/{epochs} {history}")
            if self.scheduler:
                self.scheduler.step()
        return history.history

    def set_dataloaders(self, batch_size, shuffle):
        dataloaders = {
            'inner': self.mesh.build_dataloader(batch_size, shuffle),
            'bocos': {}
        }
        for boco in self.bocos:
            dataloaders['bocos'][boco.name] = boco.build_dataloader(
                batch_size, shuffle)
        return dataloaders

    def computeGrads(self, outputs, inputs):
        grads, = torch.autograd.grad(outputs, inputs,
                                     grad_outputs=outputs.data.new(
                                         outputs.shape).fill_(1),
                                     create_graph=True, only_inputs=True)
        return grads

    def eval(self, mesh, batch_size=None):
        dataloader = mesh.build_dataloader(batch_size, shuffle=False)
        outputs = torch.tensor([]).to(mesh.device)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                outputs = torch.cat([outputs, self.model(batch)])
        return outputs
