import numpy as np

class ReLU():
    def acti(self, s):
        return np.maximum(0, s)

    def der(self, x):
        return np.where(x > 0, 1, 0)
