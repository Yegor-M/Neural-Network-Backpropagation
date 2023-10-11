from Neuron import Neuron
import numpy as np

class Layer():
  def __init__(self, input, output, acti, eta):
    self.neurons=[]
    self.input=input
    for i in range(output):
      self.neurons.append(Neuron(input, acti, eta))

  def predict(self, x):
    y=[]
    for n in self.neurons:
      y.append(n.predict(x))
    return y

  def fit(self, e):
    e_p=np.zeros(self.input)
    for i, n in enumerate(self.neurons):
      e_p_one = n.fit(e[i])
      e_p+=e_p_one
    return e_p