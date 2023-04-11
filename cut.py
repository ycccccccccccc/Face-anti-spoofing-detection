import cv2
import math
import numpy as np
import csv
import os
from os import listdir


def position_array(face_position_path):
    f = open(face_position_path, 'r')
    faces = f.readlines()
    array = []

    for face in faces:
        array.append(face)

    f.close()
    return array


def crop_face_from_scene(image_path, face_name_full, scale):
    face_position = face_name_full.split(' ')
    x1 = int(face_position[0])
    y1 = int(face_position[1])
    x2 = int(face_position[2])
    y2 = int(face_position[3])
    h = y2 - y1
    w = x2 - x1
    image = cv2.imread(image_path)
    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - h_scale / 2.0
    x1 = x_mid - w_scale / 2.0
    y2 = y_mid + h_scale / 2.0
    x2 = x_mid + w_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), h_img)
    x2 = min(math.floor(x2), w_img)
    cropped_image = image[y1:y2, x1:x2]
    image_x = cv2.resize(cropped_image, (256, 256))
    return image_x


train_live = []
path = '/home/fas1/dataset/SiW_image/Train_files/live'
fpath = '/img/SiW/Train/live'
output = '/home/fas1/Daniel/SiW/train/live'
zero = "0 0 0 0 \n"
for folder in os.listdir(path):
    output_folder = os.path.join(output, folder)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
        folder_num = folder.split('-')
        folder_face = folder + '.face'
        face_path = os.path.join(fpath, folder_num[0], folder_face)
        face_position_matrix = position_array(face_path)
        print("now cropping: " + folder)
        for image_path in os.listdir(os.path.join(path, folder)):
            if image_path.endswith('.jpg'):
                face_scale = np.random.randint(12, 15)
                face_scale = face_scale / 10.0

                image_num = image_path.split('-')
                num = image_num[5]
                num = num.replace('.jpg', '')
                position = face_position_matrix[int(num)]
                if position != zero:
                    tmp = image_path
                    image_path = os.path.join(path, folder, image_path)
                    img_x = crop_face_from_scene(image_path, position, face_scale)
                    cv2.imwrite(os.path.join(output_folder, tmp), img_x)

        print("finish croping:" + folder + "\n")
