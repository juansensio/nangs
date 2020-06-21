import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, device="cpu"):
        mesh = np.stack(np.meshgrid(*data), -1).reshape(-1, len(data))
        self.X = torch.from_numpy(mesh).float().to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, ix):
        return self.X[ix]


class Mesh():
    def __init__(self, data, device="cpu"):
        assert isinstance(data, dict), "you must pass a dict with your data"
        self.vars, data = tuple(data.keys()), data.values()
        self.dataset = Dataset(data, device)
        self.device = device

    def build_dataloader(self, batch_size=None, shuffle=True):
        if batch_size == None:
            batch_size = len(self.dataset)

        return torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle)
