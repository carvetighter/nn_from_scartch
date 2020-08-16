'''
this file finishes chapter 4 from 
Neural Networks from Scratch

added activtion functions ReLU & softmax to nn_class file
'''

'''
imports
'''

import numpy
import nnfs
from matplotlib import pyplot
from nnfs.datasets import spiral_data
from nn_class import Layer_Dense
from nn_class import Activation_ReLU
from nn_class import Activation_Softmax

# some initializations
nnfs.init()

'''
plot spiral data
'''

X, y = spiral_data(100, 3)
# pyplot.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
# pyplot.show()

# objects
dense1 = Layer_Dense(2, 3)
activtion1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# forward pass
dense1.forward(X)
activtion1.forward(dense1.output)
dense2.forward(activtion1.output)
activation2.forward(dense2.output)

# results
print('ddd dense1 output ddd')
print(dense1.output.shape)
print(dense1.output[:5], '\n')

print('ddd activation1 output ddd')
print(activtion1.output.shape)
print(activtion1.output[:5], '\n')

print('ddd dense2 output ddd')
print(dense2.output.shape)
print(dense2.output[:5], '\n')

print('ddd activation2 output ddd')
print(activation2.output.shape)
print(activation2.output[:5])