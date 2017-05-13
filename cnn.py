
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
    model.add(Convolution2D(filters = 32,
                        kernel_size = 3,
                        strides = 1,
                        padding = 'same',
                        activation = 'relu'))
                        
    model.add(MaxPooling2D(strides = 2, pool_size = 2))
    model.add(Flatten())
    
    model.add(Dense(units = 128, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model

#CREATING TEST SET AND TRAINING SET
from keras.preprocessing.image import ImageDataGenerator
batch_size = 25

train_datagen = ImageDataGenerator(rescale = 1/255,
                                   rotation_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1/255)

train_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                  batch_size = batch_size,
                                                  class_mode = 'binary',
                                                  target_size = (64,64))

test_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                  batch_size = batch_size,
                                                  class_mode = 'binary',
                                                  target_size = (64,64))
build_model().fit_generator(train_dataset,
                            steps_per_epoch = 8000/batch_size,
                            epochs = 25,
                            validation_data = test_dataset,
                            validation_steps = 2000/batch_size,
                            workers = 32,
                            max_q_size = 16)

build_model().fit_generator(test_dataset, steps_per_epoch = 2000, epochs = 1)

#TESTING ON SINGLE IMAGE
from keras.preprocessing import image
image_open = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
image_array = np.asarray(image_open, dtype="uint8")
image_array = np.expand_dims(image_array, axis = 0)

result = build_model().predict(image_array)
class_index = train_dataset.class_indices
