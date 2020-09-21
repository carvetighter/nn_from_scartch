'''
this file finishes chapter 14 from 
Neural Networks from Scratch

Notes from Chapter 14:
layer size 64 results w/o L2 regularization:
    Train:
    loss -> 0.1048
    data loss -> 0.1048
    regularization loss -> 0.0
    accuracy -> 0.9633
    learning rate -> 0.019

    Validation:
    loss -> 1.0869
    accuracy -> 0.7866

** at 64 neuron layer w/o L2 regularization the model is definately overfitting;
   validation loss is is significantly higher and accuracy is 17% lower

layer size 64 results w/ L2 regularization:
    Train:
    loss -> 0.1986
    data loss -> 0.1339
    regularization loss -> 0.0586
    accuracy -> 0.9667
    learning rate -> 0.019

    Validation:
    loss -> 0.5076
    accuracy -> 0.8433

** at 64 neuron layer w/ L2 regularization the model is overfitting less;
   capacity may be too big due to higher validation accuracy; the loss is larger
   on the validation set which suggests the model may be overfitting; increase the
   layers to 126 to see what happens

layer size 256 results w/ L2 regularization:
    Train:
    loss -> 0.1648
    data loss -> 0.1141
    regularization loss -> 0.0529
    accuracy -> 0.9700
    learning rate -> 0.019

    Validation:
    loss -> 0.4626
    accuracy -> 0.8533

** at 256 neuron layer the model w/ L2 regularization seems to confirm we are still
   overfitting with a large gap in accuracy and a lower accury for the validation set 
   and a higher loss; test 32 neuron layer

layer size 32 results w/ L2 regularization:
    Train:
    loss -> 0.3437
    data loss -> 0.2712
    regularization loss -> 0.0725
    accuracy -> 0.9000
    learning rate -> 0.019

    Validation:
    loss -> 0.5278
    accuracy -> 0.8000

** at 32 neuron layer the model w/ L2 regularization still seems to be overfitting 
   with a lower training loss, a 10% gap in accuracy and lower validation accuracy; 
   just for kicks lets try 512 neuron layer

layer size 512 results w/ L2 regularization:
    Train:
    loss -> 0.1773
    data loss -> 0.1149
    regularization loss -> 0.0624
    accuracy -> 0.9667
    learning rate -> 0.019

    Validation:
    loss -> 0.4382
    accuracy -> 0.8533

** at 512 neurons w/ L2 regularization the results were slightly worse than 256 
   in training and only slightly better than 64 neurons on both training and validation

layer size 16 results w/ L2 regularization:
    Train:
    loss -> 0.5026
    data loss -> 0.4465
    regularization loss -> 0.0560
    accuracy -> 0.8366
    learning rate -> 0.019

    Validation:
    loss -> 0.6999
    accuracy -> 0.7133
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
from nn_class import create_data

# some initializations
nnfs.init()

'''
creatre spiral data
'''

X, y = spiral_data(samples = 100, classes = 3)
X_test, y_test = spiral_data(samples = 100, classes = 3)

'''
create nn objects
'''
int_layer_size = 32
dense1 = Layer_Dense(2, int_layer_size,
    # weight_regularization_l1 = 5e-4,
    # bias_regularization_l1 = 5e-4,
    weight_regularization_l2 = 5e-4,
    bias_regularization_l2 = 5e-4
    )
activtion1 = Activation_ReLU()
dense2 = Layer_Dense(int_layer_size, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
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

    # activation / loss for dense3; takes output of dense3
    float_data_loss = loss_activation.forward(dense2.output, y)
    
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
float_test_loss = loss_activation.forward(dense2.output, y_test)

# nn performance metrics & predictions
array_y_test_pred = numpy.argmax(loss_activation.output, axis = 1)
float_test_accuracy = numpy.mean(array_y_test_pred == y_test)

# results
string_test_results = 'ddd test results -> loss = {:.5f}, accuracy = {:.5f} ddd'
print(string_test_results.format(float_test_loss, float_test_accuracy))
