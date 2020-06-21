import numpy as np


class History():
    def __init__(self, precision=5):
        self.history = {}
        self.current = {}
        self.precision = precision

    def add(self, d):
        for name, metric in d.items():
            if not name in self.history:
                self.history[name] = []
            self.history[name].append(metric)

    def add_step(self, d):
        for name, metric in d.items():
            if name not in self.current:
                self.current[name] = []
            self.current[name].append(metric)

    def average(self):
        return {name: round(np.mean(self.current[name]), self.precision) for name in self.current}

    def step(self):
        for name in self.current:
            self.add({name: np.mean(self.current[name])})
        self.current = {}

    def __str__(self):
        s = ''
        for name, value in self.history.items():
            s += f' | {name} {round(value[-1], self.precision)}'
        return s
