# ACTIVATE carnd: "activate carnd-term1"

# Import Libraries
import csv
import pickle
import numpy as np
import math
import pandas as pd
from sklearn.utils import shuffle
import os

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg

from keras.layers import Flatten, Dense, Activation, Input
from keras.models import Model, Sequential

import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer

# Fix error with TF and Keras
tf.python.control_flow_ops = tf

# Load Labels Data and Define y_train values
data=pd.read_csv('driving_log.csv', sep=',',header=None)
y_train = data.values[:,3]
# PRINT VARIABLES FOR DEBUGING
print('\nLabels Loaded\n')
#print('\nData Type: ' + str(type(data)))
#print('Data Values Type: ' + str(type(data.values)))
#print('Driving Log CSV Data Shape: ' + str(data.shape))
#print('Values: \n' + str(data.values[:3,0]))

# Load Features Data (Images) and defne X_train values
training_file = 'train.pickle'
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
X_train = train['features']
print('Train Data Loaded')
print ('X_train Data Type After Loaded with Pickle: ' + str(type(X_train)))
X_train = np.array(X_train) #converts from list to numpy.ndarray
print('X_train Lenghth: ' + str(len(X_train)))

print('\nImage Shape: ' + str(X_train[0].shape))
print('Image Type: ' + str(type(X_train[0])))

# Plot One Image
#plt.figure(figsize=(2,2))
#plt.imshow(X_train[951])
#plt.show()

# PRINT VARIABLES FOR DEBUGING
#print(X_train[25])
#print(y_train[25])
print('TYPE OF X VARIABLE: ' + str(type(X_train)))
print('TYPE OF Y VARIABLE: ' + str(type(y_train)))

# Shuffle the data
X_train, y_train = shuffle(X_train, y_train)
print('Data Shuffled')

# Normalize Data to gray scale between -0.5 and 0.5 FUNCTION
def normalize_grayscale(image_data):
	a = -0.5
	b = 0.5
	grayscale_min = 0.0
	grayscale_max = 255.0
	return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

X_normalized = normalize_grayscale(X_train)
print('Data Normalized')
#print(X_normalized[951])

#plt.figure(figsize=(2,2))
#plt.imshow(X_normalized[951])
#plt.show()

# One Hot encode the labels to the variable y_one_hot
label_binarizer = LabelBinarizer()
#y_one_hot = label_binarizer.fit_transform(y_train)
#print('ONE HOT ENCODE SHAPE: ' + str(y_one_hot.shape))
print('y_train TYPE: ' + str(type(y_train)))


# Build a Multi-Layer Feedforward Network
print('\nImage Shape: ' + str(X_train[0].shape))
model = Sequential()

model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(43))
model.add(Activation('softmax'))


# Compile and train the model here.
#model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#history = model.fit(X_normalized, y_train, nb_epoch=2, validation_split=0.2)