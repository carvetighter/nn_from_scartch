'''
this file finishes chapter 14 from 
Neural Networks from Scratch

??
'''

'''
imports
'''

import numpy
# import nnfs
from matplotlib import pyplot
# from nnfs.datasets import spiral_data
from nn_class import Layer_Dense
from nn_class import Activation_ReLU
from nn_class import Activation_Softmax
from nn_class import Loss_CategoricalCrossetropy
from nn_class import Optimizer_SGD
from nn_class import Optimizer_Adagrad
from nn_class import Optimizer_RMSprop
from nn_class import Optimizer_Adam
from nn_class import create_data

# some initializations
# nnfs.init()

'''
plot spiral data
'''

X, y = create_data(100, 3)
X_test, y_test = create_data(100, 3)

# X, y = spiral_data(100, 3)
# pyplot.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
# pyplot.show()

'''
create nn objects
'''

dense1 = Layer_Dense(2, 64)
activtion1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossetropy()
# optimizer = Optimizer_SGD(decay = 5e-8)
# optimizer = Optimizer_SGD(decay = 1e-8, momentum = 0.7)
# optimizer = Optimizer_Adagrad(decay = 1e-8)
# optimizer = Optimizer_RMSprop(decay = 1e-8)
# optimizer = Optimizer_RMSprop(learning_rate = 0.05, decay = 4e-8, rho = 0.999)
optimizer = Optimizer_Adam(learning_rate = 0.05, decay = 1e-8)
bool_verbose = False

'''
train the model
'''
for int_epoch in range(0, 10001):
    '''
    nn forward pass
    '''

    dense1.forward(X)
    activtion1.forward(dense1.output)
    dense2.forward(activtion1.output)
    activation2.forward(dense2.output)
    float_loss = loss_function.forward(activation2.output, y)

    '''
    nn performance metrics
    '''

    array_y_pred = numpy.argmax(activation2.output, axis = 1)
    float_accuracy = numpy.mean(array_y_pred == y)

    '''
    nn backward pass
    '''

    loss_function.backward(activation2.output, y)
    activation2.backward(loss_function.dvalues)
    dense2.backward(activation2.dvalues)
    activtion1.backward(dense2.dvalues)
    dense1.backward(activtion1.dvalues)

    '''
    optimize / update weights & biases
    '''

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

    '''
    results
    '''
    if int_epoch % 1000 == 0:
        print('ddd for epoch {}: loss = {:.5f} ddd'.format(int_epoch, float_loss))
        print('ddd for epoch {}: accuracy = {:.5f} ddd'.format(int_epoch, float_accuracy))
        print('ddd for epoch {}: learning rate = {:.5f} ddd\n'.format(int_epoch, optimizer.current_learning_rate))

        if bool_verbose:
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
            print(activation2.output[:5], '\n')

            print('ddd dense2 gradient weights ddd')
            print(dense2.dweights.shape)
            print(dense2.dweights, '\n')

            print('ddd dense2 gradient values ddd')
            print(dense2.dvalues.shape)
            print(dense2.dvalues[:5], '\n')

            print('ddd dense2 gradient biases ddd')
            print(dense2.dbiases.shape)
            print(dense2.dbiases, '\n')

            print('ddd dense1 gradient weights ddd')
            print(dense1.dweights.shape)
            print(dense1.dweights, '\n')

            print('ddd dense1 gradient values ddd')
            print(dense1.dvalues.shape)
            print(dense1.dvalues[:5], '\n')

            print('ddd dense1 gradient biases ddd')
            print(dense1.dbiases.shape)
            print(dense1.dbiases)

'''
test the model
'''
# nn forward pass
dense1.forward(X_test)
activtion1.forward(dense1.output)
dense2.forward(activtion1.output)
activation2.forward(dense2.output)
float_test_loss = loss_function.forward(activation2.output, y_test)

# nn performance metrics & predictions
array_y_test_pred = numpy.argmax(activation2.output, axis = 1)
float_test_accuracy = numpy.mean(array_y_test_pred == y_test)

# results
string_test_results = 'ddd test results -> loss = {:.5f}, accuracy = {:.5f} ddd'
print(string_test_results.format(float_test_loss, float_test_accuracy))