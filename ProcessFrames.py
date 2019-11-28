import cv2
import numpy as np
import os


def create_folder(folders):

    for folder in folders:

        try:
            os.mkdir(folder)
            print("Folder: " + folder + " created.")

        except FileExistsError:
            print("Folder: " + folder + " already exists.")


def subtract_frames(folder_name, sub_folder_name, file_name):

    print("Folder name: " + folder_name)
    print("File name: " + file_name)

    cap = cv2.VideoCapture(folder_name + "/" +file_name)
    print(cap)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    _, previous_frame = cap.read()

    iterations = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    for count, frame in enumerate(range(iterations)):
        previous_gray_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

        _, current_frame = cap.read()
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(current_gray_frame, previous_gray_frame)
        _, thres_frame_diff = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

        filename = file_name.split(".")[0] + "_" + str(count) + ".jpg"

        cv2.imwrite(os.path.join(sub_folder_name, file_name), thres_frame_diff)

        previous_frame = current_frame

    cap.release()
    

if __name__ == '__main__':

    actions_list = []
    subtracted_actions_list = []

    path = './data/'

    for folder in os.listdir(path):
        print(folder)
        actions_list.append(path + folder)
        subtracted_actions_list.append(path + folder + "_subtracted")

    print(actions_list)

    create_folder(subtracted_actions_list)

    videos_list = []

    for action, sub_action in zip(actions_list, subtracted_actions_list):
        videos_list = os.listdir(action)

        for video in videos_list:
            subtract_frames(action, sub_action, video)
