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

def create_data(m_int_points, m_int_classes):
    '''
    '''
    X = numpy.zeros((m_int_points * m_int_classes, 2))
    y = numpy.zeros((m_int_points * m_int_classes), dtype = 'uint8')
    for int_class_number in range(0, m_int_classes):
        ix = range(m_int_points * int_class_number, m_int_points * (int_class_number + 1))
        r = numpy.linspace(0.0, 1, m_int_points) # radius
        t = numpy.linspace(int_class_number * 4, (int_class_number + 1) * 4, m_int_points) \
            + numpy.random.randn(m_int_points) * 0.5
        X[ix] = numpy.c_[r * numpy.sin(t * 2.5), r * numpy.cos(t * 2.5)]
        y[ix] = int_class_number
    return X, y

'''
classes
'''

class Loss_CategoricalCrossetropy(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        self.float_mean_loss = None
        self.dvalues = None
    
    def forward(self, y_pred, y_true):
        '''
        method used to predict the corss entorpy loss; usually after the softmax of the 
        output layer

        :param numpy.array y_pred: array of floats by class which are the probibilites of
            each class; each row in the array is a sample; each column is a class
        :param numpy.array y_true: array of boolean ints (0, 1) which indicate the class
            is true prediction; each row in the array is a sample; each column is a class
        :rtype: float
        :return: categorical cross-entorpy loss of a network
        '''
        int_num_samples = len(y_pred)
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(0, int_num_samples), y_true]
        array_neg_log_likelihoods = -numpy.log(y_pred)
        if len(y_true.shape) == 2:
            array_neg_log_likelihoods *= y_true
        self.float_mean_loss = numpy.mean(array_neg_log_likelihoods)
        return self.float_mean_loss
    
    def backward(self, dvalues, y_true):
        '''
        '''
        int_num_samples = len(dvalues)
        self.dvalues = dvalues.copy()
        self.dvalues[range(0, int_num_samples), y_true] -= 1
        self.dvalues = self.dvalues / int_num_samples
        
class Activation_Softmax(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        # not this will be the probabilities of the activation
        self.output = None
        self.inputs = None
        self.dvalues = None
    
    def forward(self, inputs):
        '''
        '''
        self.inputs = inputs
        exp_values = numpy.exp(inputs - numpy.max(inputs, axis = 1, keepdims = True))
        self.output = exp_values / numpy.sum(exp_values, axis = 1, keepdims = True)
    
    def backward(self, dvalues):
        '''
        '''
        self.dvalues = dvalues.copy()

class Activation_ReLU(object):
    def __init__(self):
        '''
        '''
        self.inputs = None
        self.output = None
        self.dvalues = None
    
    def forward(self, inputs):
        '''
        '''
        self.inputs = inputs
        self.output = numpy.maximum(0., inputs)
    
    def backward(self, dvalues):
        '''
        '''
        self.dvalues = dvalues.copy()
        self.dvalues[self.inputs <= 0] = 0

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
        self.inputs = None
        self.dweights = None
        self.dbiases = None
        self.dbalues = None
    
    # forward pass
    def forward(self, inputs):
        '''
        '''
        self.inputs = inputs
        self.output = numpy.dot(inputs, self.weights) + self.biases
    
    # backward pass
    def backward(self, dvalues):
        '''
        '''
        self.dweights = numpy.dot(self.inputs.T, dvalues)
        self.dbiases = numpy.sum(dvalues, axis = 0, keepdims = True)
        self.dvalues = numpy.dot(dvalues, self.weights.T)

'''
apply functions
'''
