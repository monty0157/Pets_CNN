
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
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

#CREATING TEST SET AND TRAINING SET
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                  batch_size = 32,
                                                  class_mode = 'binary',
                                                  target_size = (64,64))

test_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                  batch_size = 32,
                                                  class_mode = 'binary',
                                                  target_size = (64,64))

build_model().fit_generator(train_dataset,
                            samples_per_epoch = 8000,
                            nb_epoch = 25,
                            validation_data = test_dataset,
                            nb_val_samples = 2000)