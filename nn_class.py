'''
<file doc string>
'''

'''
imports
'''

import numpy

'''
suppotive functions
'''

'''
classes
'''

class Layer_Dense(object):
    '''
    '''

    # constructor
    def __init__(self, inputs, neurons):
        '''
        '''
        self.weights = 0.01 * numpy.random.randn(inputs, neurons)
        self.biases = numpy.zeros(shape = (1, neurons))
        self.output = None
    
    # forward pass
    def forward(self, inputs):
        '''
        '''
        self.output = numpy.dot(inputs, self.weights) + self.biases

'''
apply functions
'''
