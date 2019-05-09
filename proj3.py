# CS4391.001 - Project 3
#
# This program trains a convolutional
# neural network to recognize images
# from the cfar10 dataset.
#
# Program by: Eric Busch
# edb160230@utdallas.edu

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

# set the logging verbosity to ERROR in order to get rid of warnings about deprecated versions
tf.logging.set_verbosity(tf.logging.ERROR)

# set the random seeds to make sure your results are reproducible
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

# get training/testing datasets
cifar10 = keras.datasets.cifar10
(X_train, y_train),(X_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create and compile the training model
model = keras.models.Sequential([keras.layers.Conv2D(32,(5,5),padding='same',activation=tf.nn.relu,input_shape=(32,32,3)),
                                 keras.layers.Conv2D(32,(3,3),padding='same',activation=tf.nn.relu),
                                 keras.layers.MaxPooling2D(2,2),
                                 keras.layers.Dropout(0.25),
                                 keras.layers.Conv2D(64,(5,5),padding='same',activation=tf.nn.relu),
                                 keras.layers.Conv2D(64,(3,3),padding='same',activation=tf.nn.relu),
                                 keras.layers.MaxPooling2D(2,2),
                                 keras.layers.Dropout(0.25),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(512, activation=tf.nn.relu),
                                 keras.layers.Dropout(0.5),
                                 keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model on the training dataset
print('Preprocess Training Data:')
X_train = X_train /255.0 # do preprocessing to the training data
print('Training:')
model.fit(X_train, y_train, batch_size=48, epochs=5)

# test the trained model on the testing dataset
print('Preprocess Testing Data:')
X_test = X_test /255.0 # do preprocessing to the testing data
print('Testing:')
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:{}, Test Loss:{} '.format(test_acc, test_loss)) # print results of test

# generate predictions and print results of testing
print("Predicting the class for some sample test data:")
prob_result = model.predict(X_test[0:25])
class_result = prob_result.argmax(axis = -1)
print(class_result.shape)
plt.figure("CFAR10 sample test results",figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    label = '{} as {}'.format(class_names[y_test[i,0]], class_names[class_result[i]])
    plt.xlabel(label)
plt.show()
