import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D

#Load Training Data
X_Train = np.load('X_Train_Raw.npy')
y_Train = np.load('y_Train_Raw.npy')
print(X_Train.shape)
print(y_Train.shape)

# Modified PilotNet
keep_prob = 0.5 #Dropout keep probability
batch_size = 64 #Batch size
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5,
	input_shape=(160,320,3)))
# Crop
model.add(Cropping2D(cropping=((50,25),(0,0))))
# Start conv layers
model.add(Convolution2D(24,5,5, subsample=(2,2), init='normal'))
model.add(Convolution2D(36,5,5, subsample=(2,2), init='normal'))
model.add(Convolution2D(48,5,5, subsample=(2,2), 
	init='normal', activation='elu'))
model.add(Dropout(keep_prob)) # Dropout #1
model.add(Convolution2D(64,3,3, subsample=(1,1), init='normal'))
model.add(Convolution2D(64,3,3, subsample=(1,1), 
	init='normal', activation='elu'))
model.add(Dropout(keep_prob)) # Dropout #2
# Start of fully connected layers
model.add(Flatten())
model.add(Dense(100, init='normal', activation='tanh'))
model.add(Dropout(keep_prob)) # Dropout #3
model.add(Dense(50, init='normal'))
model.add(Dense(10, init='normal'))
model.add(Dense(1, init='normal'))

# Train Model (MSE + Adam)
model.compile(loss='mse', optimizer='adam')
model.fit(X_Train, y_Train, validation_split=0.2, 
shuffle=True, nb_epoch=1, batch_size=batch_size)

model.save('model.h5')
exit()