import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, comp=-np.inf):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.comp = comp

    def early_stop_score(self, score):
        if score > self.comp:
            self.comp = score
            self.counter = 0
        elif score <= (self.comp - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_loss(self, loss):
        if loss < self.comp:
            self.comp = loss
            self.counter = 0
        elif loss >= (self.comp + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
