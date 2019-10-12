from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

dim = 512 

def load_dataset(img_width = dim, img_heigth = dim):
    
    path = 'data/'
    
    folders = os.listdir(path)

    X = []
    Y = []

    print(folders)

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

                X.append(new_frame.astype("float16")/255.)
                Y.append(folder)

    X = np.array(X).reshape((-1, img_width, img_heigth, 1))

    return X, Y
def compare_results(frames, predicted, n_frames = 10):

    columns = 2
    fig = plt.figure(figsize=(10,10))

    for i in range(1, columns * n_frames + 1):
        print(i)
        fig.add_subplot(n_frames, columns, i)

        if i % 2 == 1:
            plt.imshow(frames[i, : , : , 0], cmap = 'gray')
        else:
            plt.imshow(predicted[i, : , : , 0], cmap = 'gray')
    
    plt.show()
    fig.savefig('resultado.png', dpi = fig.dpi)


def load_model():
    json_model = open('model.json', 'r')
    json_model_loaded = json_model.read()
    json_model.close()

    loaded_model = model_from_json(json_model_loaded)

    loaded_model.load_weights('model.h5')
    print('Loaded model from disk')

    return loaded_model

if __name__ == "__main__":

    x, y = load_dataset()
    print('(x, y)')
    print(np.shape(x), np.shape(y))


    loaded_model = load_model()

    while key is not 0:
        n_frames = np.random.radint(len(x))
        reconstructed_frames = loaded_model.predict(x[n_frames])

        compare_results(x, reconstructed_frames, n_frames=1)
