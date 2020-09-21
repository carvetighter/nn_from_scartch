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
    def __init__(self, num_inputs, num_neurons, weight_regularization_l1 = 0.,
        weight_regularization_l2 = 0., bias_regularization_l1 = 0.,
        bias_regularization_l2 = 0.):
        '''
        '''
        # initial weight & bias
        self.weights = 0.01 * numpy.random.randn(num_inputs, num_neurons)
        self.biases = numpy.zeros(shape = (1, num_neurons))

        # regularization
        self.weight_regularization_l1 = weight_regularization_l1
        self.weight_regularization_l2 = weight_regularization_l2
        self.bias_regularization_l1 = bias_regularization_l1
        self.bias_regularization_l2 = bias_regularization_l2

        # deltas
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

        # inputs / output
        self.output = None
        self.inputs = None
        
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
        # gradients on paramaters
        self.dweights = numpy.dot(self.inputs.T, dvalues)
        self.dbiases = numpy.sum(dvalues, axis = 0, keepdims = True)
        
        # gradients on regularization
        # l1 on weights
        if self.weight_regularization_l1 > 0.:
            dl1 = self.weights.copy()
            dl1[dl1 >= 0] = 1
            dl1[dl1 < 0] = -1
            self.dweights += self.weight_regularization_l1 * dl1
        
        # l2 on weights
        if self.weight_regularization_l2 > 0.:
            self.dweights += 2 * self.weight_regularization_l2 * self.weights
        
        if self.bias_regularization_l1 > 0.:
            dl1 = self.biases.copy()
            dl1[dl1 >= 0] = 1
            dl1[dl1 < 0] = -1
            self.dbiases += self.bias_regularization_l1 * dl1
        
        if self.bias_regularization_l2 > 0.:
            self.dbiases += 2 * self.bias_regularization_l2 * self.biases

        # gradients on values
        self.dinputs = numpy.dot(dvalues, self.weights.T)

class Activation_ReLU(object):
    def __init__(self):
        '''
        '''
        self.inputs = None
        self.output = None
        self.dinputs = None
    
    def forward(self, inputs):
        '''
        '''
        self.inputs = inputs
        self.output = numpy.maximum(0., inputs)
    
    def backward(self, dvalues):
        '''
        '''
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        # not this will be the probabilities of the activation
        self.output = None
        self.inputs = None
        self.dinputs = None
    
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
        # set-up
        self.dinputs = dvalues.copy()

        # enumerate outputs & gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(1, -1)

            # calculate jacobian matrix of the output & 
            jacobian_matrix = numpy.diagflat(single_output) - \
                numpy.dot(single_output, single_output.T)
            
            # calculate sample-wise gradient
            self.dinputs[index] = numpy.dot(jacobian_matrix, single_dvalues)

class Loss(object):
    '''
    base class for loss calculations
    '''
    def __init__(self):
        '''
        '''
        self.float_regularization_loss = 0.
        self.float_data_loss = 0.
    
    def regularization_loss(self, layer):
        '''
        '''
        # set-up
        self.float_regularization_loss = 0

        # l1 regularization (weights); calculate only when factor > 0
        if layer.weight_regularization_l1 > 0.:
            self.float_regularization_loss += layer.weight_regularization_l1 * \
                numpy.sum(numpy.abs(layer.weights))
        
        # l2 regularization (weights)
        if layer.weight_regularization_l2 > 0.:
            self.float_regularization_loss += layer.weight_regularization_l2 * \
                numpy.sum(layer.weights * layer.weights)
        
        # l1 regularization (biases); calculate when factor > 0
        if layer.bias_regularization_l1 > 0.:
            self.float_regularization_loss += layer.bias_regularization_l1 * \
                numpy.sum(numpy.abs(layer.biases))
        
        # l2 regularization (biases)
        if layer.bias_regularization_l2 > 0.:
            self.float_regularization_loss += layer.bias_regularization_l2 * \
                numpy.sum(layer.biases * layer.biases)
        
        return self.float_regularization_loss
    
    def calculate(self, output, y):
        '''
        '''
        # calculate sample loss
        sample_loss = self.forward(output, y)

        # calculate mean loss
        self.float_data_loss = numpy.mean(sample_loss)

        return self.float_data_loss
    
class Loss_CategoricalCrossetropy(Loss):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(Loss_CategoricalCrossetropy, self).__init__()
        self.float_mean_loss = None
        self.dinputs = None
    
    def forward(self, y_pred, y_true):
        '''
        method used to predict the corss entorpy loss; usually after the softmax of the 
        output layer

        :param numpy.array y_pred: array of floats by class which are the probibilites of
            each class; each row in the array is a sample; each column is a class
        :param numpy.array y_true: array of boolean ints (0, 1) which indicate the class
            is true prediction; each row in the array is a sample; each column is a class
        :rtype: numpy.array
        :return: categorical cross-entorpy loss of a network
        '''
        # number of sample in batch
        int_num_samples = len(y_pred)

        # clip data to prevent division by 0; clip both sides to not drag mean towards
        # any value
        y_pred_clipped = numpy.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values (only if categorical labels)
        if len(y_true.shape) == 1:
            y_pred_clipped = y_pred_clipped[range(0, int_num_samples), y_true]

        # losses
        array_neg_log_likelihoods = -numpy.log(y_pred_clipped)

        # mask values (only for on-hot encoded labels)
        if len(y_true.shape) == 2:
            array_neg_log_likelihoods *= y_true
        
        return array_neg_log_likelihoods
    
    def backward(self, dvalues, y_true):
        '''
        '''
        int_num_samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(0, int_num_samples), y_true] -= 1
        self.dinputs = self.dinputs / int_num_samples

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
            self.current_learning_rate = self.learning_rate * \
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
            self.current_learning_rate = self.learning_rate * \
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
            self.current_learning_rate = self.learning_rate * \
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
            self.current_learning_rate = self.learning_rate * \
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

class Activation_Softmax_Loss_CategoricalCrossentropy():
    '''
    '''
    
    def __init__(self):
        '''
        constructor
        '''
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossetropy()
        self.output = None
        self.dinputs = None
    
    def forward(self, inputs, y_true):
        '''
        '''
        # output layer's activtion function
        self.activation.forward(inputs)

        # set ouput
        self.output = self.activation.output

        # calculate and return loss value
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        '''
        '''
        # number of samples
        num_samples = len(dvalues)

        # calculate gradient
        self.dinputs = dvalues.copy()
        self.dinputs[range(0, num_samples), y_true] -= 1

        # normalize gradient
        self.dinputs = self.dinputs / num_samples

'''
apply functions
'''
