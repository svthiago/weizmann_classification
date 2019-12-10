import pandas as pd
from glob2 import glob
from keras.models import load_model


if __name__ == "__main__":

    names = ["daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"]

    for i, name in enumerate(names):
        load_name = './data/*/' + name + '_*'
        frames = glob(load_name)

        model = load_model('./classifier_models/classifier_' + str(i) + '.h5')

