import os
import re

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob2 import glob
from loguru import logger
from natsort import natsorted
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import load_model, Model, Sequential


def load_dataset(path, sorted_names, img_width, img_heigth):
    
    folders = sorted(os.listdir(path), key=str)

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


                elif re.match(eval_name, name):
                    eval_data.append(new_frame.astype("float32")/255.)
                    eval_labels.append(folder)

    # X = np.array(X).reshape((-1, img_width, img_heigth, 1))
    train_data = np.array(train_data).reshape((-1, img_width, img_heigth, 1))
    eval_data = np.array(eval_data).reshape((-1, img_width, img_heigth, 1))

    train_labels = pd.get_dummies(train_labels).values
    eval_labels = pd.get_dummies(eval_labels).values

    return train_data, train_labels, eval_data, eval_labels

if __name__ == "__main__":
    models = glob('./models/*.h5')
    models = sorted(models, key = str)
    print(models)

    num_classes = 10
    # dim = 256
    dim = 32
    n_epochs = 50
    batch_size = 4

    logger.add('file_{time}.log')

    path = './data/'

    names = ["daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"]

    set_list = {'train_names': [], 'eval_name': []}



    for i, model in enumerate(models):
        autoencoder = load_model(model)

        del autoencoder.layers[-9:]
        print(autoencoder.summary())

        encoder = autoencoder.layers[-1].output

        model = Sequential(autoencoder.layers)

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate = 0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

        print('###########')

        print(model.summary())

        set_list['eval_name'] = names.pop()
        set_list['train_names'] = names
        logger.debug(set_list)

        # omitting the labels
        x_train, y_train, x_eval, y_eval = load_dataset(path, set_list, dim, dim)

        # inserting the last name into head of the list
        names.insert(0, set_list['eval_name'])


        # # checking data proportions
        total_len = len(x_train) + len(x_eval)

        logger.debug("Total len: " + str(total_len))

        logger.debug('train data len: ' + str(len(x_train)))
        logger.debug('eval_data: '+ str(len(x_eval)))

        logger.debug('train data(%): ' + str(len(x_train) / total_len))
        logger.debug('eval_data(%): ' + str(len(x_eval) / total_len))
        logger.debug('###################')

        ## train the model
        model.fit(x_train, y_train,
                batch_size=1,
                epochs=10,
                verbose=1,
                validation_data=(x_eval, y_eval))

        del x_train
        del y_train
        del x_eval
        del y_eval

        model.save('./models/classifier_' + str(i) + '.h5')