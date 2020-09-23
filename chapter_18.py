'''
this file finishes chapter 17 from 
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

Chapter 18 notes:
implementing the Model class
'''

'''
imports
'''

import numpy
import nnfs
from matplotlib import pyplot
from nnfs.datasets import spiral_data
from nnfs.datasets import sine_data

# class imports
from nn_class import Layer_Dense
from nn_class import Layer_Dropout
from nn_class import Activation_ReLU
from nn_class import Activation_Softmax
from nn_class import Activation_Sigmoid
from nn_class import Activation_Linear
from nn_class import Loss_CategoricalCrossetropy
from nn_class import Loss_BinaryCrossentropy
from nn_class import Loss_MeanSquaredError
from nn_class import Activation_Softmax_Loss_CategoricalCrossentropy
from nn_class import Optimizer_SGD
from nn_class import Optimizer_Adagrad
from nn_class import Optimizer_RMSprop
from nn_class import Optimizer_Adam
from nn_class import Model
from nn_class import Accuracy_Regression
from nn_class import create_data

# some initializations
nnfs.init()

'''
creatre sine data
'''

X, y = sine_data()
X_test, y_test = sine_data()

'''
create model & set-up
'''

int_layer_size = 64
model = Model()

'''
add model layers, loss, optimizer, accuracy, finalize
'''

# add layers
model.add(Layer_Dense(1, int_layer_size))
model.add(Activation_ReLU())
model.add(Layer_Dense(int_layer_size, int_layer_size))
model.add(Activation_ReLU())
model.add(Layer_Dense(int_layer_size, 1))
model.add(Activation_Linear())

# set loss, optimizer, accuracy
model.set(
    loss = Loss_MeanSquaredError(),
    optimizer = Optimizer_Adam(learning_rate = 0.005, decay = 1e-3),
    accuracy = Accuracy_Regression()
)

# finalize
model.finalize()

'''
train the model
'''

model.train(X, y, epochs = 10000, print_every = 1000, validation_data = (X_test, y_test))

'''
test the model
'''
'''
# nn forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
activation3.forward(dense3.output)
float_test_loss = loss_function.calculate(activation3.output, y_test)

# nn performance metrics & predictions
array_y_test_pred = activation3.output
float_test_accuracy = numpy.mean(
    numpy.absolute(array_y_test_pred - y_test) < accuracy_precision
)

# results
string_test_results = 'ddd test results -> loss = {:.5f}, accuracy = {:.5f} ddd'
print(string_test_results.format(float_test_loss, float_test_accuracy))

# plot test results
pyplot.plot(X_test, y_test)
pyplot.plot(X_test, activation3.output)
pyplot.show()
'''
