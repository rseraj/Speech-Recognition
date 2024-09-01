
"""
Created on Sat Jun 13 22:24:00 2020

@author: seraj
"""
from PIL import Image	
from matplotlib import pyplot
import os
import sys
import numpy as np
import cv2
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import adam
#from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.models import Model

def read_data(path):
  images=[]
  labels = []
  listpath = os.listdir(path)
  for folder in listpath:
      filelist = os.listdir(os.path.join(path, folder))
      for file in filelist:
          samples = cv2.imread(path + "/" + folder + "/" + file)
          images.append(samples)
          labels.append(int(folder))
  images = np.array(images)
  return images, labels

path_train="E:/tam5/TrainSpecResize"
path_test="E:/tam5/TestSpecResize"

train_images, train_labels = read_data(path_train)
test_images, test_labels = read_data(path_train)

#print("train_images[199] shape: ", train_images[199].shape)
#pyplot.imshow(train_images[199])
#print("trainX[1] type: ", trainX[1].dtype)

# convert from integers to floats
train = train_images.astype('float32')
test = test_images.astype('float32')

# normalize to range 0-1

X_train = train/255.0
X_test = test/255.0

# one hot encode target values
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)


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
    


    
model = load_model('mymodel.hdf5')
model.pop()
model.summary()
model.add(Dense(10,activation='softmax'))


model.summary()

# compile model





model.compile(optimizer=adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
#==================================================
# simple early stopping
es = EarlyStopping(monitor='val_loss',min_delta=1e-3,patience=5,verbose=1,mode='auto'
                   ,restore_best_weights=True )
# Train our model
network_history = model.fit(X_train, Y_train, batch_size=32#,verbose=2,
, callbacks=[es] ,epochs=50, validation_split= .2 ,shuffle=True) 


plot_history(network_history)

# Evaluation

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test accuracy:', score[1]*100) 


test_labels_p = model.predict(X_test)

test_labels_p = np.argmax(test_labels_p, axis=1)
from keras.models import load_model



model.save('mymodelOnSpecImg.hdf5')
print("Saved model to disk")



       