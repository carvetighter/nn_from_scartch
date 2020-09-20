'''
this file finishes chapter 14 from 
Neural Networks from Scratch

??
'''

'''
imports
'''

import numpy
import nnfs
#from matplotlib import pyplot
from nnfs.datasets import spiral_data

# class imports
from nn_class import Layer_Dense
from nn_class import Activation_ReLU
from nn_class import Activation_Softmax
from nn_class import Loss_CategoricalCrossetropy
from nn_class import Activation_Softmax_Loss_CategoricalCrossentropy
from nn_class import Optimizer_SGD
from nn_class import Optimizer_Adagrad
from nn_class import Optimizer_RMSprop
from nn_class import Optimizer_Adam
# from nn_class import create_data

# some initializations
nnfs.init()

'''
creatre spiral data
'''

# X, y = create_data(100, 3)
# X_test, y_test = create_data(100, 3)

X, y = spiral_data(samples = 100, classes = 3)
X_test, y_test = spiral_data(samples = 100, classes = 3)
# pyplot.scatter(X[:, 0], X[:, 1], c = y, cmap = 'brg')
# pyplot.show()

'''
create nn objects
'''

dense1 = Layer_Dense(2, 64, weight_regularization_l2 = 5e-4,
    bias_regularization_l2 = 5e-4)
activtion1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
activation2 = Activation_Softmax()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# loss_function = Loss_CategoricalCrossetropy()
# optimizer = Optimizer_SGD(decay = 5e-8)
# optimizer = Optimizer_SGD(decay = 1e-8, momentum = 0.7)
# optimizer = Optimizer_Adagrad(decay = 1e-8)
# optimizer = Optimizer_RMSprop(decay = 1e-8)
# optimizer = Optimizer_RMSprop(learning_rate = 0.05, decay = 4e-8, rho = 0.999)
optimizer = Optimizer_Adam(learning_rate = 0.02, decay = 5e-7)

'''
train the model
'''
for int_epoch in range(0, 10001):
    '''
    nn forward pass
    '''
    # 1st forward pass of dense1
    dense1.forward(X)

    # activation function for dense1; takes output of dense1
    activtion1.forward(dense1.output)

    # foward pass of dense2; takes output of activation1
    dense2.forward(activtion1.output)

    # activation / loss for dense2; takes output of dense2
    float_data_loss = loss_activation.forward(dense2.output, y)
    # activation2.forward(dense2.output)
    # float_data_loss = loss_function.forward(activation2.output, y)
    
    # calculate regularization loss
    float_regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
        loss_activation.loss.regularization_loss(dense2)
    
    # calculate total loss
    float_loss = float_data_loss + float_regularization_loss

    '''
    nn performance metrics
    '''

    array_y_pred = numpy.argmax(loss_activation.output, axis = 1)
    float_accuracy = numpy.mean(array_y_pred == y)

    '''
    nn backward pass
    '''

    loss_activation.backward(loss_activation.output, y)
    # loss_function.backward(activation2.output, y)
    # activation2.backward(loss_function.dinputs)
    dense2.backward(loss_activation.dinputs)
    activtion1.backward(dense2.dinputs)
    dense1.backward(activtion1.dinputs)

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
        print('ddd for epoch {}: data loss = {:.5f} ddd'.format(int_epoch, float_data_loss))
        print('ddd for epoch {}: regularization loss = {:.5f} ddd'.format(int_epoch, float_regularization_loss))
        print('ddd for epoch {}: accuracy = {:.5f} ddd'.format(int_epoch, float_accuracy))
        print('ddd for epoch {}: learning rate = {:.5f} ddd\n'.format(int_epoch, optimizer.current_learning_rate))

'''
test the model
'''

# nn forward pass
dense1.forward(X_test)
activtion1.forward(dense1.output)
dense2.forward(activtion1.output)
# activation2.forward(dense2.output)
float_test_loss = loss_activation.forward(dense2.output, y_test)

# nn performance metrics & predictions
array_y_test_pred = numpy.argmax(loss_activation.output, axis = 1)
float_test_accuracy = numpy.mean(array_y_test_pred == y_test)

# results
string_test_results = 'ddd test results -> loss = {:.5f}, accuracy = {:.5f} ddd'
print(string_test_results.format(float_test_loss, float_test_accuracy))
