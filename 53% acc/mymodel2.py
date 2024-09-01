# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 19:44:47 2020

@author: seraj
"""

import os
import sys
import numpy as np
#from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
#from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.models import load_model

def plot_history(net_history):
    history = net_history.history
    import matplotlib.pyplot as plt
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['acc']
    val_accuracies = history['val_acc']
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])
    
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['acc', 'val_acc'])
    
 
# load train and test dataset

(train_images, train_labels), (test_images, test_labels) =cifar10.load_data()

# Data attributes
print("train_images dimentions: ", train_images.ndim)
print("train_images shape: ", train_images.shape)
print("train_images type: ", train_images.dtype)
print("test_images shape: ", test_images.shape)


# convert from integers to floats
train = train_images.astype('float32')
test = test_images.astype('float32')
# normalize to range 0-1
X_train = train/255.0
X_test = test/255.0

# one hot encode target values
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)

# creating mymodel
model = Sequential()
model.add(Conv2D(7,(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(9,(3,3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10,activation='softmax'))
model.summary()
# compile model

model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
#==================================================
# simple early stopping
es = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=5,verbose=1,mode='auto'
                   ,restore_best_weights=True )
# Train our model
network_history = model.fit(X_train, Y_train, batch_size=32,#verbose=2
                           callbacks=[es] ,epochs=1000, validation_split= .2 ,shuffle=True) 
                            
plot_history(network_history)

# Evaluation

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test accuracy:', score[1]*100) 


test_labels_p = model.predict(X_test)

test_labels_p = np.argmax(test_labels_p, axis=1)




model.save('mymodel.hdf5')
print("Saved model to disk")

