import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from Layer import Layer

class NeuronNetwork():
  def __init__(self, layers, acti, eta):
    self.layers=[]
    for i in range(1, len(layers)):
      self.layers.append(Layer(layers[i-1], layers[i], acti, eta))
  
  def predict(self, x):
    x_in=x.copy()
    for l in self.layers:
      x_in=l.predict(x_in)
    return x_in

  def fit(self, e):
    layers_reverse = self.layers.copy()
    layers_reverse.reverse()
    for l in layers_reverse:
      e = l.fit(e)


import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from Layer import Layer  # Assuming you have a Layer class

class NeuronNetwork():
    def __init__(self, layers, acti, eta):
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i - 1], layers[i], acti, eta))

    def predict(self, x):
        x_in = x.copy()
        for l in self.layers:
            x_in = l.predict(x_in)
        return x_in

    def fit(self, e):
        layers_reverse = self.layers.copy()
        layers_reverse.reverse()
        for l in layers_reverse:
            e = l.fit(e)