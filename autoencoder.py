import os
import cv2
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D

dim = 320

def load_dataset(img_width = dim, img_heigth = dim):
    path = "data/"
    
    folders = os.listdir(path)

    X = []
    Y = []

    for folder in folders:

        if os.path.isdir(path + folder):
            frames = os.listdir(path + folder)
            print("Loading: " + folder)

            for frame in frames:
                # load a new frame
                new_frame = cv2.imread(path + folder + "/" + frame)

                # convert to grayscale
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2GRAY)
                # resize frame
                new_frame = cv2.resize(new_frame, (img_width, img_heigth))

                X.append(new_frame.astype("float32")/255.)
                Y.append(folder)

    X = np.array(X).reshape((-1, img_width, img_heigth, 1))

    return X, Y



def to_one_hot_encoding(labels):

    labels = np.array(labels)

    onehot_encoder = OneHotEncoder(sparse = False)
    onehot_encoded = labels.reshape(len(labels), 1)
    onehot_encoded = onehot_encoder.fit_transform(onehot_encoded)

    return onehot_encoded

def getModel(img_width = dim, img_heigth = dim):

    input_img = Input(shape=(img_width, img_heigth, 1))
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(input_img)
    x = MaxPooling2D((2, 2), padding='same', dim_ordering='tf')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    x = MaxPooling2D((2, 2), padding='same', dim_ordering='tf')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    x = MaxPooling2D((2, 2), padding='same', dim_ordering='tf')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    encoded = MaxPooling2D((2, 2), padding='same', dim_ordering='tf')(x)


    #6x6x32 -- bottleneck

    x = UpSampling2D((2, 2), dim_ordering='tf')(encoded)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    x = UpSampling2D((2, 2), dim_ordering='tf')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    x = UpSampling2D((2, 2), dim_ordering='tf')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    x = UpSampling2D((2, 2), dim_ordering='tf')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)
    decoded = Convolution2D(1, 3, 3, activation='relu', border_mode='same', dim_ordering='tf')(x)

    #Create model
    autoencoder = Model(input_img, decoded)

    return autoencoder


def trainModel():
    # Load dataset
    print("Loading dataset...")


    X, Y = load_dataset()

    # Y = to_one_hot_encoding(Y)

    print(X[-1], Y[-1])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3) # , random_state = 23)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size = 0.3) #, random_state = 23)


    print('#'*10)
    print(np.shape(x_train))
    print('#'*10)

    # Create model description
    print("Creating model...")
    model = getModel()
    model.compile(optimizer='adadelta', loss='mean_squared_error') #, metrics=['accuracy'])

    # Train model
    print("Training model...")
    model.fit(x_train, x_train, nb_epoch=10, batch_size=20, shuffle=False, validation_data=(x_test, x_test), callbacks=[TensorBoard(log_dir='/tmp/tb', histogram_freq=0, write_graph=False)])

    # Evaluate loaded model on test data
    print("Evaluating model...")
    score = model.evaluate(x_val, x_val, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1] , score[1] * 100))
    print("%s: %.2f%%" % (model.metrics_names , score * 100))

    # Serialize model to JSON
    print("Saving model...")
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    print("Saving weights...")

    model.save_weights("model.h5")

    return model, x_val


def compare_results(frames, predicted, n_frames = 10):

    columns = 2
    fig = plt.figure(figsize=(10,10))

    for i in range(1, columns * n_frames + 1):
        fig.add_subplot(n_frames, columns, i)

        if i % 2 == 1:
            plt.imshow(frames[i, : , : , 0], cmap = 'gray')
        else:
            plt.imshow(predicted[i, : , : , 0], cmap = 'gray')
    
    plt.show()
    fig.savefig('resultado.png', dpi = fig.dpi)

# this method load all frame names within the action folders
def load_filenames(path):

    folders = os.listdir(path)

    persons = []

    for folder in folders:
        person_dict = defaultdict()
        if os.path.isdir(path + folder): # folder is the name of the action
            files = os.listdir(path + folder)

            for file in files:
                print(file.split('_')[0])
                break
        break
                            

if __name__ == "__main__":

    # these settings are default
    # img_width = 48
    # img_heigth = 36

    load_filenames('data')

    X, Y = load_dataset()
    #
    Y = to_one_hot_encoding(Y)
    #
    # print(X[-1], Y[-1])
    #
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3) #, random_state = 23)

###

    model, x_val = trainModel()
    reconstructed_frames = model.predict(x_val)

    compare_results(x_val, reconstructed_frames)

###

    plt.imshow(reconstructed_frames[1, : , : , 0 ])
    # plt.imshow(x_val[1, : , : , 0])
    plt.show()
