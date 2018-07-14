import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
#from keras.backend import tf as ktf

#Load Training Data
X_Train = np.load('X_Train_Raw.npy')
y_Train = np.load('y_Train_Raw.npy')
print(X_Train.shape)
print(y_Train.shape)

# Simple Starting Model
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, 
# input_shape=(160,320,3)))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Flatten())
# model.add(Dense(1))

# WORKING!!!!!
# # NVIDIA CNN
# model = Sequential()
# # Normalization
# model.add(Lambda(lambda x: x / 255.0 - 0.5, 
	# input_shape=(160,320,3)))
# # Crop
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# #model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(3, nrows,ncols)))
# model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
# model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
# model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
# model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
# model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
# model.add(Flatten())
# model.add(Dense(1164, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='tanh'))

# NVIDIA CNN
keep_prob = 0.5
batch_size = 64
model = Sequential()
# Normalization
model.add(Lambda(lambda x: x / 255.0 - 0.5,
	input_shape=(160,320,3)))
# Crop
model.add(Cropping2D(cropping=((50,25),(0,0))))
# Resize
#model.add(Lambda(lambda x: ktf.image.resize_images(x, (66,200))))
#model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(3, nrows,ncols)))
model.add(Convolution2D(24,5,5, subsample=(2,2), init='normal'))
model.add(Convolution2D(36,5,5, subsample=(2,2), init='normal'))
model.add(Convolution2D(48,5,5, subsample=(2,2), 
	init='normal', activation='elu'))
model.add(Dropout(keep_prob))
model.add(Convolution2D(64,3,3, subsample=(1,1), init='normal'))
model.add(Convolution2D(64,3,3, subsample=(1,1), 
	init='normal', activation='elu'))
model.add(Dropout(keep_prob))
model.add(Flatten())
#model.add(Dense(1164, init='normal'))
model.add(Dense(100, init='normal', activation='tanh'))
model.add(Dropout(keep_prob))
model.add(Dense(50, init='normal'))
model.add(Dense(10, init='normal'))
model.add(Dense(1, init='normal'))

# Train Model
model.compile(loss='mse', optimizer='adam')
model.fit(X_Train, y_Train, validation_split=0.2, 
shuffle=True, nb_epoch=1, batch_size=batch_size)

model.save('model.h5')
exit()