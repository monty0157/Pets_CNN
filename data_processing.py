import os
from keras.preprocessing import image
import random
import numpy as np

#PREPARING DATA FOR GRID SEARCH
def grid_search_helper(target_size):
    cats_path_train = './dataset/training_set/cats/'
    cats_list_train = os.listdir(cats_path_train)
    dogs_path_train = './dataset/training_set/dogs/'
    dogs_list_train = os.listdir(dogs_path_train)

    images_list = []
    labels_list = []
    for file in cats_list_train:
        if (file != '.DS_Store'):
            img = image.load_img(cats_path_train + file, target_size = target_size)
            img = np.asarray(img)

            #RESCALE IMAGE
            img = img/255
            images_list.append(img)

            #ADD AS CLASS 0
            labels_list.append(0)
    for file in dogs_list_train:
        if (file != '.DS_Store'):
            img = image.load_img(dogs_path_train + file, target_size = target_size)
            img = np.asarray(img)

            #RESCALE IMAGE
            img = img/255
            images_list.append(img)

            #ADD AS CLASS 1
            labels_list.append(1)

    #SHUFFLE DATA
    zip_data_for_shuffle = list(zip(images_list,labels_list))
    random.shuffle(zip_data_for_shuffle)
    images_list, labels_list = zip(*zip_data_for_shuffle)

    return np.asarray(images_list), np.asarray(labels_list)
