import numpy as np

class Sigmoid():
  def acti(self, s):
    return 1 / (1 + np.exp(-s))
  def der(self, x):
    a = self.acti(x)
    return a * (1 - a)