import numpy as np

class Arctang():
    def acti(self, s):
        return np.arctan(s)

    def der(self, x):
        return 1 / (1 + x**2)