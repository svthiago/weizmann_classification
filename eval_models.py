import pandas as pd
from glob2 import glob
from keras.models import load_model


def load_by_person(load_name, img_width, img_heigth):
    
    folders = sorted(glob(path), key=str)


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



if __name__ == "__main__":

    names = ["daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"]

    for i, name in enumerate(names):
        load_name = './data/*/' + name + '_*'
        frames = glob(load_name)

        model = load_model('./classifier_models/classifier_' + str(i) + '.h5')

