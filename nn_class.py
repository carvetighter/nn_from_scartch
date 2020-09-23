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

class Model(object):
    '''
    '''
    def __init__(self):
        '''
        constructor
        '''
        self.layers = list()
        self.loss = None
        self.optimizer = None
        self.input_layer = None
        self.output_layer_activation = None
        self.trainable_layers = list()
        self.accuracy = None
    
    def add(self, layer):
        '''
        adds laye4rs to the model
        '''
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy):
        '''
        set loss and optimizer

        note:
        * -> makes the arguements following required key word arguements; will need 
             to name and set values explicitly
        '''
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def train(self, X, y, *, epochs = 1, print_every = 1, validation_data = None):
        '''
        '''
        # inialize accuracy
        self.accuracy.init(y)

        # main training loop
        for epoch in range(1, epochs + 1):
            # forward pass
            output = self.forward(X)

            # calculate loss
            data_loss, regularization_loss = self.loss.calculate(output, y,
                include_regularization_loss = True)
            train_loss = data_loss + regularization_loss

            # get predictions and caclulate accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # backward pass
            self.backward(output, y)

            # optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # print summary
            if not epoch % print_every:
                print('ddd for epoch {}: loss = {:.5f} ddd'.format(
                    epoch, train_loss)
                )
                print('ddd for epoch {}: data loss = {:.5f} ddd'.format(
                    epoch, data_loss)
                )
                print('ddd for epoch {}: regularization loss = {:.5f} ddd'.format(
                    epoch, regularization_loss)
                )
                print('ddd for epoch {}: accuracy = {:.5f} ddd'.format(
                    epoch, accuracy)
                )
                print('ddd for epoch {}: learning rate = {:.5f} ddd\n'.format(
                    epoch, self.optimizer.current_learning_rate)
                )

        # if there is validation data
        if validation_data is not None:
            # split tuple
            X_val, y_val = validation_data

            # forward pass
            output = self.forward(X_val)

            # calculate loss
            val_loss = self.loss.calculate(output, y_val)

            # get predictions
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # print test results
            string_test_results = 'ddd test results -> loss = {:.5f}, accuracy = {:.5f} ddd'
            print(string_test_results.format(val_loss, accuracy))

    def finalize(self):
        '''
        '''
        # create and set input layer
        self.input_layer = Layer_Input()

        # count objects in layers
        layer_count = len(self.layers)

        # iterate of objects
        for i in range(0, layer_count):
            # if first layer the previous called object will be the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[ i - 1]
                self.layers[i].next = self.layers[i + 1]
            # if last layer the next layer is loss
            # need to save reference to last object which is the model output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
            
            # if layer has attribute "weights" then it's iterable
            # add to list of iterable layers
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
            
            # update loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)
            
    def forward(self, X):
        '''
        '''
        # call forward method for input layer; this will set our output property that
        # the first layer is expecting in the 'prev' object
        self.input_layer.forward(X)

        # call forward method of every object in the chain; pass output of the previous
        # object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output)
        
        # return the last layer from the list; in this case would be the last activation
        # layer
        return layer.output

    def backward(self, output, y):
        '''
        '''
        # first call backward method of loss; this will set dinputs property of the
        # last layer
        self.loss.backward(output, y)

        # call backward method of through all objects in reverse order passing dinputs
        # as the parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
        
class Accuracy(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        pass
    
    def calculate(self, predictions, y):
        '''
        '''
        # get comparison results
        comparisons = self.compare(predictions, y)

        # calculate accuracy
        accuracy = numpy.mean(comparisons)

        # return accuracy
        return accuracy

class Accuracy_Regression(Accuracy):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(Accuracy_Regression, self).__init__()
        self.precision = None
    
    def init(self, y, reinit = False):
        '''
        '''
        if self.precision is None or reinit:
            self.precision = numpy.std(y) / 250
    
    def compare(self, predictions, y):
        '''
        '''
        return numpy.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(Accuracy_Categorical, self).__init__()
    
    def init(self, y):
        '''
        '''
        pass
    
    def compare(self, predictions, y):
        '''
        '''
        return predictions == y

class Layer_Input(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        self.output = None
    
    def forward(self, inputs):
        '''
        '''
        self.output = inputs

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
        self.weights = 0.1 * numpy.random.randn(num_inputs, num_neurons)
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

class Layer_Dropout(object):
    '''
    '''
    def __init__(self, rate):
        '''
        constructor
        '''
        self.rate = 1 - rate
        self.inputs = None
        self.output = None
        self.dinputs = None
        self.binary_mask = None
    
    def forward(self, inputs):
        '''
        '''
        # generate and save scaled mask
        self.inputs = inputs
        self.binary_mask = numpy.random.binomial(1, self.rate, size = inputs.shape) \
            / self.rate
        
        # apply mask to output values
        self.output = self.binary_mask * inputs
    
    def backward(self, dvalues):
        '''
        '''
        # gradient on values
        self.dinputs = dvalues * self.binary_mask

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

    def predictions(self, outputs):
        '''
        '''
        return outputs

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

    def predictions(self, outputs):
        '''
        '''
        return numpy.argmax(outputs, axis = 1)

class Activation_Sigmoid(object):
    '''
    '''
    def __init__(self):
        '''
        '''
        self.inputs = None
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        '''
        '''
        # calcualte sigmoid of inputs
        self.inputs = inputs
        self.output = 1/ (1 + numpy.exp(-inputs))
    
    def backward(self, dvalues):
        '''
        '''
        # deriviative, calculates from output of sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    def predictions(self, outputs):
        '''
        '''
        return (outputs > 0.5) * 1

class Activation_Linear(object):
    '''
    '''
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
        self.output = inputs
    
    def backward(self, dvalues):
        '''
        '''
        # derivitive
        self.dinputs = dvalues.copy()

    def predictions(self, outputs):
        '''
        '''
        return outputs

class Loss(object):
    '''
    base class for loss calculations
    '''
    def __init__(self):
        '''
        '''
        self.float_regularization_loss = 0.
        self.float_data_loss = 0.
        self.trainable_layers = None
    
    def remember_trainable_layers(self, trainable_layers):
        '''
        remember trainable layers
        '''
        self.trainable_layers = trainable_layers

    def regularization_loss(self):
        '''
        '''
        # set-up
        self.float_regularization_loss = 0.

        # l1 regularization (weights); calculate only when factor > 0
        for layer in self.trainable_layers:
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
    
    def calculate(self, output, y, *, include_regularization_loss = False):
        '''
        '''
        # calculate sample loss
        sample_loss = self.forward(output, y)

        # calculate mean loss
        self.float_data_loss = numpy.mean(sample_loss)

        if not include_regularization_loss:
            return self.float_data_loss
        
        return self.float_data_loss, self.regularization_loss()
    
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
        method used to predict the cross entorpy loss; usually after the softmax of the 
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

        # mask values (only for one-hot encoded labels)
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

class Loss_BinaryCrossentropy(Loss):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(Loss_BinaryCrossentropy, self).__init__()
        self.dinputs = None

    def forward(self, y_pred, y_true):
        '''
        '''
        # clip data to prevent divide by zero
        y_pred_clipped = numpy.clip(y_pred, 1e-7, 1 - 1e-7)

        # calculate sample-wise loss
        sample_loss = -(y_true * numpy.log(y_pred_clipped) + 
            (1 - y_true) * numpy.log(1 - y_pred_clipped))
        sample_loss = numpy.mean(sample_loss, axis = -1)

        return sample_loss
    
    def backward(self, dvalues, y_true):
        '''
        '''
        # number of samples
        n_samples = len(dvalues)

        # clip to prevent divide by zero
        clipped_dvalues = numpy.clip(dvalues, 1e-7, 1 - 1e-7)

        # caclulate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1  - clipped_dvalues))
        self.dinputs = self.dinputs / n_samples        

class Loss_MeanSquaredError(Loss):
    '''
    '''
    def __init__(self):
        '''
        '''
        super(Loss_MeanSquaredError, self).__init__()
        self.dinputs = None

    def forward(self, y_pred, y_true):
        '''
        '''
        # calcualte and return loss
        sample_loss = numpy.mean((y_true - y_pred)**2, axis = -1)
        return sample_loss
    
    def backward(self, dvalues, y_true):
        '''
        '''
        # number of samples
        n_samples = len(dvalues)

        # calculate gradient
        self.dinputs = -2 * (y_true - dvalues)

        # normalize gradient
        self.dinputs = self.dinputs / n_samples

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

        # set output
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
