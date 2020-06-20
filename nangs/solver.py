from fastprogress import master_bar, progress_bar
from .history import History
import torch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Solver():

    def compile(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

    def computePDELoss(self, vars, grads):
        print("This function need to be overloaded !!!")

    def solve(self, mesh, bocos, epochs=50, batch_size=None, shuffle=True):
        dataloaders = self.set_dataloaders(mesh, bocos, batch_size, shuffle)
        # solve PDE
        history = History()
        mb = master_bar(range(1, epochs+1))
        for epoch in mb:
            history.add('lr', get_lr(self.optimizer))
            # iterate over the internal points in batches
            for batch in progress_bar(dataloaders['inner'], parent=mb):
                X = batch
                self.optimizer.zero_grad()
                # optimize for boundary points
                for boco in bocos:
                    for batch in dataloaders['bocos'][boco.name]:
                        loss = boco.computeLoss(
                            batch, self.model, self.criterion)
                        loss.backward()
                        history.add_step(boco.name, loss.item())
                # optimize for internal points
                X.requires_grad = True
                p = self.model(X)
                loss = self.computePDELoss(X, p)
                loss.backward()
                self.optimizer.step()
                history.add_step('PDE', loss.item())
                mb.child.comment = str(history.average())
            history.step()
            mb.write(f"Epoch {epoch}/{epochs} {history}")
            if self.scheduler:
                self.scheduler.step()
        return history.history

    def set_dataloaders(self, mesh, bocos, batch_size, shuffle):
        dataloaders = {
            'inner': mesh.build_dataloader(batch_size, shuffle),
            'bocos': {}
        }
        names = []
        for boco in bocos:
            assert boco.name not in names, "Each boco needs its own name to avoid conflits"
            names.append(boco.name)
            dataloaders['bocos'][boco.name] = boco.build_dataloader(
                batch_size, shuffle)
        return dataloaders

    def eval(self, mesh, batch_size=None):
        dataloader = mesh.build_dataloader(batch_size, shuffle=False)
        outputs = torch.tensor([]).to(mesh.device)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                outputs = torch.cat([outputs, self.model(batch)])
        return outputs
