import numpy as np

class Neuron():
  def __init__(self, input, acti, eta):
    self.W = np.random.uniform(-1, 1, size=(input,))
    self.bias = np.random.uniform(-1, 1)
    self.acti=acti
    self.eta=eta

  def predict(self, x):
    y=self.W*x
    y=sum(y)
    y+=self.bias
    self.last_s=y
    return self.acti.acti(y) 

  def fit(self, e):
    e_p=self.W*e 
    d=self.acti.der(self.last_s)*e
    self.W -=self.W+self.eta*d
    self.bias -=self.bias+self.eta*d
    return e_p