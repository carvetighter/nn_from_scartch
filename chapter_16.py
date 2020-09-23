'''
this file finishes chapter 16 from 
Neural Networks from Scratch

Chapter 14 notes (regularization L1 & L2):

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

Chapter 15 notes (dropout):

layer size 32 w/ regularization, droput @ 10%:
    Train:
    loss -> 0.77809
    data loss -> 0.7627
    regularization loss -> 0.00153
    accuracy -> 0.6633
    learning rate -> 0.019

    Validation:
    loss -> 0.7623
    accuracy -> 0.6666

** at 32 neurons w/ regulariation, dropout @ 10% the accuracy decreased significantly and
   the loss increased significantly; the good aspect is that our validation set is 
   slightly better than our training performance even if the accuracy is 66%; the 
   network is not overfitting; we can scale up the layers to see how the nework performs

layer size 64 w/ regularization, droput @ 10%:
    Train:
    loss -> 0.6639
    data loss -> 0.6351
    regularization loss -> 0.0287
    accuracy -> 0.7033
    learning rate -> 0.019

    Validation:
    loss -> 0.6346
    accuracy -> 0.7066

** at 64 neurons w/ regularization, dropout @ 10% the accuracy is better @ 70% and not
   overfitting; let's increase to 256 neurons

layer size 256 w/ regularization, dropout @ 10%:
    Train:
    loss -> 0.4451
    data loss -> 0.3663
    regularization loss -> 0.0787
    accuracy -> 0.8600
    learning rate -> 0.019

    Validation:
    loss -> 0.3654
    accuracy -> 0.8666

** at 256 neurons w/ regularization, dropout @ 10% the accuracy is better @ ~87%; let's
   try 512 to see it it will get over 90

layer size 512 w/ regularization, dropout @ 10%:
    Train:
    loss -> 0.3448
    data loss -> 0.2591
    regularization loss -> 0.0857
    accuracy -> 0.9100
    learning rate -> 0.019

    Validation:
    loss -> 0.2581
    accuracy -> 0.9100

** at 512 neurons w/ regularization, dropout @ 10% we hot our target, 91%; there are
   still signs of overfitting; the validation accuracy is close to the training accuracy
   (exactly the same), the validation accuracy is usually higher and the validation 
   loss os lower than expected
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
from nn_class import Layer_Dropout
from nn_class import Activation_ReLU
from nn_class import Activation_Softmax
from nn_class import Activation_Sigmoid
from nn_class import Loss_CategoricalCrossetropy
from nn_class import Loss_BinaryCrossentropy
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

# X, y = spiral_data(samples = 100, classes = 3)
# X_test, y_test = spiral_data(samples = 100, classes = 3)

X, y = spiral_data(samples = 100, classes = 2)
X_test, y_test = spiral_data(samples = 100, classes = 2)
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

'''
create nn objects
'''
int_layer_size = 64
dense1 = Layer_Dense(2, int_layer_size,
    # weight_regularization_l1 = 5e-4,
    # bias_regularization_l1 = 5e-4,
    weight_regularization_l2 = 5e-4,
    bias_regularization_l2 = 5e-4
    )
activation1 = Activation_ReLU()
# dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(int_layer_size, 1)
activation2 = Activation_Sigmoid()
loss_function = Loss_BinaryCrossentropy()
# loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
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
    activation1.forward(dense1.output)

    # dropout layer; takes output of activation1
    # dropout1.forward(activation1.output)

    # foward pass of dense2; takes output of dropout1
    dense2.forward(activation1.output)

    # foward pass of activation 2; takes output of dense2
    activation2.forward(dense2.output)

    # loss; data loss
    float_data_loss = loss_function.calculate(activation2.output, y)
    
    # loss; rgularization loss
    float_regularization_loss = loss_function.regularization_loss(dense1) + \
        loss_function.regularization_loss(dense2)
    
    # activation / loss for dense3; takes output of dense3
    # float_data_loss = loss_activation.forward(dense2.output, y)
    
    # calculate regularization loss
    # float_regularization_loss = loss_activation.loss.regularization_loss(dense1) + \
    #     loss_activation.loss.regularization_loss(dense2) 
    
    # calculate total loss
    float_loss = float_data_loss + float_regularization_loss

    '''
    nn performance metrics
    '''

    # array_y_pred = numpy.argmax(loss_activation.output, axis = 1)
    array_y_pred = (activation2.output > 0.5) * 1
    float_accuracy = numpy.mean(array_y_pred == y)

    '''
    nn backward pass
    '''

    # loss_activation.backward(loss_activation.output, y)
    loss_function.backward(activation2.output, y)
    # dense2.backward(loss_activation.dinputs)
    activation2.backward(loss_function.dinputs)
    dense2.backward(activation2.dinputs)
    # dropout1.backward(dense2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

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
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
float_test_loss = loss_function.calculate(dense2.output, y_test)

# nn performance metrics & predictions
# array_y_test_pred = numpy.argmax(loss_activation.output, axis = 1)
array_y_test_pred = (activation2.output > 0.5) * 1
float_test_accuracy = numpy.mean(array_y_test_pred == y_test)

# results
string_test_results = 'ddd test results -> loss = {:.5f}, accuracy = {:.5f} ddd'
print(string_test_results.format(float_test_loss, float_test_accuracy))
