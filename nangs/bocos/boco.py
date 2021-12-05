import torch


class Boco():
    def __init__(self, name):
        self.name = name

    def validate(self):
        assert self.computeLoss, "You need to specify a function to compute the loss"
