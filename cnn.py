
#IMPORT LIBRARIES
import keras
import numpy as np
import pandas as pd

#BUILD CNN
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(optimizer = 'adam', units = 128, filters = 32, kernel_size = 3, dropout_layers = 1):
    model = Sequential()
    model.add(Convolution2D(filters = filters,
                            kernel_size = kernel_size,
                            strides = 1,
                            padding = 'same',
                            activation = 'relu',
                            input_shape = (64, 64, 3)))

    model.add(MaxPooling2D(strides = 2, pool_size = 2))
    model.add(Convolution2D(filters = filters,
                        kernel_size = kernel_size,
                        strides = 1,
                        padding = 'same',
                        activation = 'relu'))

    model.add(MaxPooling2D(strides = 2, pool_size = 2))
    model.add(Flatten())

    model.add(Dense(units = units, activation = 'relu'))
    if(dropout_layers >= 1):
        model.add(Dropout(rate = 0.2))
    model.add(Dense(units = units, activation = 'relu'))

    if(dropout_layers >= 2):
        model.add(Dropout(rate = 0.2))
    model.add(Dense(units = 1, activation = 'sigmoid'))

    model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model

#CREATING TEST SET AND TRAINING GENERATORS
from keras.preprocessing.image import ImageDataGenerator
batch_size = 25
model = build_model()

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


'''#TRAINING AND EVALUATING ON IMAGE GENERATOR
model.fit_generator(train_dataset,
                            steps_per_epoch = 8000/batch_size,
                            epochs = 25,
                            validation_data = test_dataset,
                            validation_steps = 2000/batch_size,
                            workers = 32,
                            max_q_size = 16)

print('Accuracy, loss:', model.evaluate_generator(test_dataset, steps = 2000))'''

#GRIDSEARCH
from data_processing import grid_search_helper
images_list, labels_list = grid_search_helper(target_size = (64,64))

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
grid_search = KerasClassifier(build_fn = build_model)

parameters = {
    'dropout_layers': [0, 1, 2],
    'epochs': [25, 50],
    'units': [75, 200],
    'batch_size': [25, 32],
}

grid_search = GridSearchCV(estimator = grid_search, param_grid = parameters, scoring = 'accuracy', cv = 3)
grid_search = grid_search.fit(images_list, labels_list)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print('best_parameters:', best_parameters, 'Accuracy:', best_accuracy)
parameters = grid_search.cv_results_
print(parameters)

#TESTING ON SINGLE IMAGE
from keras.preprocessing import image
image_open = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
image_array = np.asarray(image_open, dtype="uint8")
image_array = np.expand_dims(image_array, axis = 0)

result = model.predict(image_array)
class_index = train_dataset.class_indices
print(result, class_index)
