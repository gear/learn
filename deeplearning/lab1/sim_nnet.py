#!/home/hoangnt/anaconda3/bin/python

# Coding utf-8
# Author: Hoang NT
# Date: 2016/10/12
# Problem 1: Back Propagation - Write your own neural network

import numpy as np
import random as r

def main():
    print('Initializing neural net...')

def sigmoid(X):
    return [np.exp(x) for x in X]

def d_sigmoid(X):
    'Derivative of sigmoid function. Numpy arrays.'
    return sigmoid(X) * (1-sigmoid(X))

def matmul(X,Y):
    'Matrix multiplication X * Y.'
    return np.matmul(X,Y)

class nnet:
    'Simple 1-D input, 1-D output neural feed-forward network'

    def __init__(self, input_shape, hidden_layers, output_shape):
        'Initialize network configurations.'
        assert len(hidden_layers) > 0, 'More than one hidden layer is required.'
        assert all(nl > 0 for nl in hidden_layers)
        assert input_shape > 0, 'Input dimensionality must be more than one.'
        assert output_shape > 0, 'Output dimensionality must be more than one.'
        assert hidden_layers[-1] == output_shape, 'Output must match.'
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.output_shape = output_shape
        self.built = False
        self.act = list()
        self.forward_saves = None
        self.gradient_saves = list()
        self.batch_size = None
        self.variables = None

    def act(self, layer_idx):
        'Return the activation function for a layer'
        idx = layer_idx % len(self.act)
        return self.act[idx]

    def build(self, activation_functions=[{'f':sigmoid, 'df':d_sigmoid}]):
        'Build network as configured.' 
        assert type(activation_functions) is list
        assert all('f' in la for la in activation_functions)
        assert all('df' in la for la in activation_functions)
        assert not self.built 
        self.variables = list()
        prev = self.input_shape
        for nl in self.hidden_layers:
            trainable = np.ndarray(shape=(prev,nl), dtype=np.float32)
            # Randomly initialize weights
            for i in range(prev):
                for j in range(nl):
                    trainable[i][j] = r.gauss(mu=0.0, sigma=1.0)
            self.variables.append(trainable)
            prev = nl
        self.input = np.ndarray(shape=self.input_shape, dtype=np.float32)
        self.output = self.variables[-1]
        self.act.extend(activation_functions)
        self.built = True

    def forward(self, data):
        'Compute the forward pass and store intermediate values.'
        if self.batch_size is None:
            self.batch_size = len(data)
        assert self.batch_size == len(data) 
        self.forward_saves = list()
        for sample in data:
            o = np.array(sample, dtype=np.float32)
            I = list()
            for weight in self.variables:
                o = matmul(o, weight)
                I.append(o)
            self.forward_saves.append(I)


    def backpropagation(self, labels):
        'Compute gradient of error with respect to each weight.'
        


if __name__ == '__main__':
    main() 
