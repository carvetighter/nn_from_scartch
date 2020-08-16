'''
this file finishes chapter 3 from 
Neural Networks from Scratch
'''

'''
imports
'''

import numpy
import nnfs
from matplotlib import pyplot
from nnfs.datasets import spiral_data
from nn_class import Layer_Dense

nnfs.init()

'''
simple test three neurons, two layers
'''

# objects
array_inputs = numpy.array(
    [[1, 2, 3, 2.5],
    [2., 5., -1., 2],
    [-1.5, 2.7, 3.3, -0.8]]
)
array_weights = numpy.array(
    [[0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]
)
array_biases = numpy.array([2., 3., 0.5])
array_weights_2 = numpy.array(
    [[0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]]
)
array_biases_2 = numpy.array([-1, 2, -0.5])

# forward pass
array_layer1_outputs = numpy.dot(array_inputs, array_weights.T) + array_biases
array_layer2_outputs = numpy.dot(array_layer1_outputs, array_weights_2.T) + array_biases_2

'''
output
'''
# print(array_layer2_outputs)

'''
plot spiral data
'''

X, y = spiral_data(100, 3)
# pyplot.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
# pyplot.show()

dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])
