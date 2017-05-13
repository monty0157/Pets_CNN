
#IMPORT LIBRARIES
import keras
import numpy as np

#BUILD CNN
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

def build_model():
    model = Sequential()
    model.add(Convolution2D(filters = 32,
                            kernel_size = 3,
                            strides = 1,
                            padding = 'same',
                            activation = 'relu',
                            input_shape = (64, 64, 3)))
    
    model.add(MaxPooling2D(strides = 2, pool_size = 2))
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    return model

