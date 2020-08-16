'''
this file finishes chapter 2 from 
Neural Networks from Scratch
'''

'''
imports
'''

import numpy

'''
simple test; one neuron; one layer
'''

array_inputs = numpy.array([1, 2, 3])
array_weights = numpy.array([0.2, 0.8, -0.5])
float_bias = 2.

output = numpy.dot(array_inputs, array_weights.T) + float_bias

'''
simple test three nuerons; one layer
'''

array_inputs = numpy.array([1, 2, 3, 2.5])

array_weights_01 = numpy.array([0.2, 0.8, -0.5, 1])
array_weights_02 = numpy.array([0.5, -0.91, 0.26, -0.5])
array_weights_03 = numpy.array([-0.26, -0.27, 0.17, 0.87])

float_bias_01 = 2.
float_bias_02 = 3.
float_bias_03 = 0.5

output = [
    # Neuron 1
    numpy.dot(array_inputs, array_weights_01.T) + float_bias_01,

    # Neuron 2
    numpy.dot(array_inputs, array_weights_02.T) + float_bias_02,

    # Neuron 3
    numpy.dot(array_inputs, array_weights_03.T) + float_bias_03
]

'''
simple test three neurons, one layer
'''

array_inputs = numpy.array([1, 2, 3, 2.5])
array_weights = numpy.array(
    [[0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]
)
array_biases = numpy.array([2., 3., 0.5])

output = numpy.dot(array_inputs, array_weights.T) + array_biases

'''
output
'''
print(output)