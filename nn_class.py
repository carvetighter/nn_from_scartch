'''
<file doc string>
'''

'''
imports
'''

import numpy
import random

'''
set-up
'''

random.seed(0)
numpy.random.seed(0)

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
            + numpy.random.randn(m_int_points) * 0.2
        X[ix] = numpy.c_[r * numpy.sin(t * 2.5), r * numpy.cos(t * 2.5)]
        y[ix] = int_class_number
    return X, y

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
        prob = exp_values / numpy.sum(exp_values, axis = 1, keepdims = True)
        self.output = prob

    def backward(self, dvalues):
        '''
        '''
        self.dvalues = dvalues.copy()

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
        int_num_samples = y_pred.shape[0]
        if len(y_true.shape) == 1:
            y_pred = y_pred[range(0, int_num_samples), y_true]
        array_neg_log_likelihoods = -numpy.log(y_pred)
        if len(y_true.shape) == 2:
            array_neg_log_likelihoods *= y_true
        self.float_mean_loss = numpy.sum(array_neg_log_likelihoods) / int_num_samples
        return self.float_mean_loss
    
    def backward(self, dvalues, y_true):
        '''
        '''
        int_num_samples = dvalues.shape[0]
        self.dvalues = dvalues.copy()
        self.dvalues[range(0, int_num_samples), y_true] -= 1
        self.dvalues = self.dvalues / int_num_samples
 
class Optimizer_SGD(object):
    '''
    '''

    def __init__(self, learning_rate = 1.0, decay = 0., momentum = 0.):
        '''
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    def pre_update_params(self):
        '''
        '''
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        '''
        '''
        # if does not have momemtum matricies add it
        if not hasattr(layer, 'weight_momentums'):
            layer.weight_momentums = numpy.zeros_like(layer.weights)
            layer.bias_momentums = numpy.zeros_like(layer.biases)
        
        # use momentum
        if self.momentum:
            # weight updates
            weight_updates = (self.momentum * layer.weight_momentums) - \
                (self.current_learning_rate * layer.dweights)
            layer.weight_momentums = weight_updates

            # bias updates
            bias_updates = (self.momentum * layer.bias_momentums) - \
                (self.current_learning_rate * layer.dbiases)
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases
        
        layer.weights += weight_updates
        layer.biases += bias_updates
    
    def post_update_params(self):
        '''
        '''
        self.iterations += 1

class Optimizer_Adagrad(object):
    '''
    '''

    def __init__(self, learning_rate = 1.0, decay = 0., episilon = 1e-7):
        '''
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = episilon
    
    def pre_update_params(self):
        '''
        '''
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        '''
        '''
        # if does not have momemtum matricies add it
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_cache = numpy.zeros_like(layer.biases)
        
        # update cache
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # gradient descent update w/ normailzation w/ square of cache
        layer.weights += -self.current_learning_rate * layer.dweights / \
            (numpy.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
            (numpy.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        '''
        '''
        self.iterations += 1

class Optimizer_RMSprop(object):
    '''
    '''

    def __init__(self, learning_rate = 0.001, decay = 0., rho = 0.9, episilon = 1e-7):
        '''
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = episilon
        self.rho = rho
    
    def pre_update_params(self):
        '''
        '''
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        '''
        '''
        # if does not have momemtum matricies add it
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_cache = numpy.zeros_like(layer.biases)
        
        # update cache
        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        # gradient descent update w/ normailzation w/ square of cache
        layer.weights += -self.current_learning_rate * layer.dweights / \
            (numpy.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / \
            (numpy.sqrt(layer.bias_cache) + self.epsilon)
    
    def post_update_params(self):
        '''
        '''
        self.iterations += 1

class Optimizer_Adam(object):
    '''
    '''

    def __init__(self, learning_rate = 0.001, decay = 0., episilon = 1e-7, beta_1 = 0.9,
        beta_2 = 0.999):
        '''
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = episilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def pre_update_params(self):
        '''
        '''
        if self.decay:
            self.current_learning_rate = self.current_learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        '''
        '''
        # if does not have momemtum matricies add it
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = numpy.zeros_like(layer.weights)
            layer.weight_cache = numpy.zeros_like(layer.weights)
            layer.bias_momentums = numpy.zeros_like(layer.biases)
            layer.bias_cache = numpy.zeros_like(layer.biases)
        
        # update momentum
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases
        
        # corrected momentums
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1**(self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1**(self.iterations + 1))
        
        # update cache w/ squared gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * layer.dbiases**2
        
        # get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))

        # update weights & biases
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (numpy.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (numpy.sqrt(bias_cache_corrected) + self.epsilon)
    
    def post_update_params(self):
        '''
        '''
        self.iterations += 1

'''
apply functions
'''
