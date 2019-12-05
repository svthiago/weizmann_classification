from collections import defaultdict
from math import ceil
import random
import sys
import os
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from natsort import natsorted
import matplotlib.pyplot as plt

from keras.models import Model
from keras.models import model_from_json
from keras.callbacks import TensorBoard, EarlyStopping
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, UpSampling2D


def load_dataset(path, sorted_names, img_width, img_heigth):
    
    folders = natsorted(os.listdir(path))

    # regex for detection of the names belonging to each dataset 
    train_names = re.compile('|'.join(sorted_names['train_names']))
    eval_name = re.compile('|'.join(sorted_names['eval_name']))

    train_data = []
    train_labels = []
    
    eval_data = []
    eval_labels = []


    logger.debug("Loading dataset")

    for folder in tqdm(folders):
        if os.path.isdir(path + folder):
            frames = natsorted(os.listdir(path + folder))

            for frame in frames:
                # load a new frame
                new_frame = cv2.imread(path + folder + "/" + frame)

                # convert to grayscale
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
                # resize frame
                new_frame = cv2.resize(new_frame, (img_width, img_heigth))

                # X.append(new_frame.astype("float32")/255.)
                # Y.append(folder)

                name = frame.split('_')[0]

                if re.match(train_names, name):
                    train_data.append(new_frame.astype("float32")/255.)
                    train_labels.append(folder)

                    flipped_frame = cv2.flip(new_frame.astype("float32")/255., 1)
                    train_data.append(flipped_frame)
                    train_labels.append(folder)


                elif re.match(eval_name, name):
                    eval_data.append(new_frame.astype("float32")/255.)
                    eval_labels.append(folder)

                    flipped_frame = cv2.flip(new_frame.astype("float32")/255., 1)
                    train_data.append(flipped_frame)
                    train_labels.append(folder)

    # X = np.array(X).reshape((-1, img_width, img_heigth, 1))
    train_data = np.array(train_data).reshape((-1, img_width, img_heigth, 1))
    eval_data = np.array(eval_data).reshape((-1, img_width, img_heigth, 1))

    train_labels = pd.get_dummies(train_labels).values
    eval_labels = pd.get_dummies(eval_labels).values

    return train_data, train_labels, eval_data, eval_labels


def sort_names(names, set_list):
    _names = names.copy()

    for _ in names:
        for set_index, set_tuple in enumerate(set_list):
            if _names:
                data_set, ratio_list, names_list = set_tuple
                rand_index = random.randrange(0, len(_names))
                ratio = len(names_list) / len(names)

                if ratio_list >= ratio:
                    set_list[set_index][2].append(_names[rand_index])
                    _names.pop(rand_index)
            else:
                continue

    return set_list


# this method load all frame names within the action folders
def split_by_person(path, frames, labels, names, sorted_names):

    # regex for detection of the names belonging to each dataset 
    train_names = re.compile('|'.join(sorted_names[0][2]))
    test_names = re.compile('|'.join(sorted_names[1][2]))
    eval_names = re.compile('|'.join(sorted_names[2][2]))

    train_data = []
    train_labels = []
    
    test_data = []
    test_labels = []

    eval_data = []
    eval_labels = []

    # for frame, label, name in zip(frames, labels, names):
    #     if re.match(train_names, name):
    #         train_data.append(frame)
    #         train_labels.append(label)
    #     elif re.match(test_names, name):
    #         test_data.append(frame)
    #         test_labels.append(label)
    #     elif re.match(eval_names, name):
    #         eval_data.append(frame)
    #         eval_labels.append(label)

    for _, frame in zip(tqdm(range(len(frames))), frames):
        for label in labels:
            for name in names:
                if re.match(train_names, name):
                    train_data.append(frame)
                    train_labels.append(label)
                elif re.match(test_names, name):
                    test_data.append(frame)
                    test_labels.append(label)
                elif re.match(eval_names, name):
                    eval_data.append(frame)
                    eval_labels.append(label)

    frames = None
    labels = None

    del frames
    del labels

    # logger.debug(len(train_data), len(test_data), len(eval_data))

    train_data = np.array(train_data, copy = False)
    test_data = np.array(test_data, copy = False)
    eval_data = np.array(eval_data, copy = False)

    return train_data, train_labels, test_data, test_labels, eval_data, eval_labels

def get_model(img_width, img_heigth):

    input_img = Input(shape=(img_width, img_heigth, 1))
    x = Conv2D(8, (5, 5), activation='relu', padding ='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (5, 5), activation='relu', padding ='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding ='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding ='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    #6x6x32 -- bottleneck

    x = UpSampling2D((2, 2))(encoded)
    x = Conv2D(64, (3, 3), activation='relu', padding ='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding ='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding ='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (5, 5), activation='relu', padding ='same')(x)
    decoded = Conv2D(1, (3, 3), activation='relu', padding ='same')(x)

    #Create model
    autoencoder = Model(input_img, decoded)

    return autoencoder


def train_model(model, train_data, eval_data, n_epochs, batch_size):

    # Create model description
    logger.debug("Creating model...")
    model.trainable = True
    model.compile(optimizer='adadelta', loss='mean_squared_error')

    # Simple early stopping
    es = EarlyStopping(monitor='val_loss', verbose=1, patience=10)

    # Train model
    logger.debug("Training model...")
    history = model.fit(train_data, train_data, epochs = n_epochs, batch_size = batch_size, verbose = 1,  shuffle=False, validation_data=(eval_data, eval_data), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=True)]) #, es])

    for layer in model.layers:
        layer.trainable = False

    # model.trainable = False

    #model.save("model.h5")

    return model, history 

def load_model(json_model, model_weights):
    json_model = open(model_json, 'r')
    json_model_loaded = json_model.read()
    json_model.close()

    loaded_model = model_from_json(json_model_loaded)

    loaded_model.load_weights(model_weights)
    logger.debug('Loaded model from disk')

    return loaded_model

def compare_results(frames, predicted, n_frames = 5):

    columns = 2
    fig = plt.figure(figsize=(50,50))

    for i in range(0, 2 * n_frames ):
        for j in range(0, columns):        
            fig.add_subplot(2 * n_frames, columns, i + 1)

            if j % 2 == 0:
                print('#'*10)
                print('frame plotted, index:', i)
                plt.imshow(frames[i, : , : , 0], cmap = 'gray')
            else:
                print('%'*10)
                print('predicted frame plotted, index:', i)
                plt.imshow(predicted[i, : , : , 0], cmap = 'gray')

    plt.show()
    fig.savefig('resultado.png', dpi = fig.dpi)

def save_model_performance(path, model_name, history):

    fig = plt.figure()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name)

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend(['train', 'val'], loc='upper left')
    fig.savefig(path, dpi = fig.dpi)


if __name__ == "__main__":

    dim = 512
    n_epochs = 50
    batch_size = 4

    # dim = 32
    # n_epochs = 4
    # batch_size = 1

    logger.add('file_{time}.log')

    path = './data/'

    names = ["daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"]

    set_list = {'train_names': [], 'eval_name': []}

    for i in range(9):
        set_list['eval_name'] = names.pop()
        set_list['train_names'] = names
        logger.debug(set_list)

        # omitting the labels
        train_data, _, eval_data, _ = load_dataset(path, set_list, dim, dim)

        # inserting the last name into head of the list
        names.insert(0, set_list['eval_name'])


        # # checking data proportions
        total_len = len(train_data) + len(eval_data)

        logger.debug("Total len: " + str(total_len))

        logger.debug('train data len: ' + str(len(train_data)))
        logger.debug('eval_data: '+ str(len(eval_data)))

        logger.debug('train data(%): ' + str(len(train_data) / total_len))
        logger.debug('eval_data(%): ' + str(len(eval_data) / total_len))
        logger.debug('###################')

        ## create the model and train it
        autoencoder = get_model(dim, dim)
        autoencoder, history = train_model(autoencoder, train_data, eval_data, n_epochs, batch_size)

        performance_graph_path = './models/autoencoder_' + str(i) + '.png'
        model_performance_name = 'Autoencoder ' + str(i)

        save_model_performance(performance_graph_path, model_performance_name, history)

        del train_data
        del eval_data

        trained_model_path = './models/autoencoder_' + str(i) + '.h5'
        autoencoder.save(trained_model_path)

#     # omitting the labels
#     train_data, _, test_data, _, eval_data, _ = split_by_person(path, frames, labels, names, sorted_names)

#     # create the model and train it
#    autoencoder = get_model(dim, dim)
#    _, history = train_model(autoencoder, train_data, eval_data, n_epochs, batch_size)

#    predicted_frames = autoencoder.predict(eval_data[:10], batch_size = 1)

#    compare_results(eval_data, predicted_frames, 5)
