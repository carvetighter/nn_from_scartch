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

class Activation_Softmax(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        # not this will be the probabilities of the activation
        self.output = None
    
    def forward(self, inputs):
        '''
        '''
        exp_values = numpy.exp(inputs - numpy.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / numpy.sum(exp_values, axis = 1, keepdims = True)

class Activation_ReLU(object):
    def __init__(self):
        '''
        '''
        self.output = None
    
    def forward(self, inputs):
        '''
        '''
        self.output = numpy.maximum(0., inputs)

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
