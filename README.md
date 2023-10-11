# Neural Network with BackPropagation

## How to train a supervised Neural Network?
1. Feed Forward: The input data is passed through the neural network layers to make predictions.
2. BackPropagation: The backpropagation algorithm, a cornerstone of training multi-layer neural networks, is used to adjust the network's parameters (weights and biases).
3. Update Weights: After calculating gradients during backpropagation, the weights and biases of the neural network are updated to minimize the error between predictions and actual outputs.

## Description of BackPropagation
Backpropagation is the implementation of gradient descent in multi-layer neural networks. Since the same training rule recursively exists in each layer of the neural network, we can calculate the contribution of each weight to the total error inversely from the output layer to the input layer, which is so-called backpropagation.

### Gradient Descent (Optimization)
Gradient descent is the primary optimization algorithm used in this project. It's an iterative method that seeks local or global minima of a function. The key idea is as follows:

1. Start from a point on the graph of a function.
2. Determine the direction ▽F(a) in which the function decreases most rapidly.
3. Move in this direction by a small step (γ) to reach a new point (a+1).

### Stochastic Gradient Descent (SGD)
The advantage of this method is that the gradient is accurate and the function converges fast. But when the training dataset is enormous, the evaluation of the gradient from all data points becomes expensive and the training time can be very long.

## Coding Description
Activation Function: Sigmoid
Error Minimization: Gradient Descent
Error Function: Mean Square Error
