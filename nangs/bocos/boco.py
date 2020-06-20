import torch


class Boco():
    def __init__(self, name):
        self.name = name
        self.dataset = None

    def build_dataloader(self, batch_size=None, shuffle=True):
        if batch_size == None:
            batch_size = len(self.dataset)

        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle)

    def computeLoss(self, batch, model, criterion):
        print("This function needs to be overloaded by a boco !!!")
