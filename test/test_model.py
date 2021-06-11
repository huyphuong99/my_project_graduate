import csv
import shutil
from builtins import int

import pandas
from recognition.model_ocr import Model
import cv2
import glob
import numpy as np
import os
import time

vocab_DAY = ['.', '-', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(img_h_DAY, img_w_DAY, max_len_DAY) = (40, 210, 10)
path_model_BIRTHDAY = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/DATE_010621_ver3_99.35%.h5"
model_BIRTHDAY = Model(vocab_DAY, img_w_DAY, img_h_DAY, max_len_DAY, path_model_BIRTHDAY)

vocab_ADDRESS = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0',
                 '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
                 '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
                 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'À',
                 'Á', 'Â', 'Ã', 'È', 'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'à',
                 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'Ă',
                 'ă', 'Đ', 'đ', 'Ĩ', 'ĩ', 'Ũ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư', 'Ạ', 'ạ', 'Ả', 'ả', 'Ấ',
                 'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ', 'Ậ', 'ậ', 'Ắ', 'ắ', 'Ằ', 'ằ', 'Ẳ', 'ẳ', 'Ẵ',
                 'ẵ', 'Ặ', 'ặ', 'Ẹ', 'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ', 'Ế', 'ế', 'Ề', 'ề', 'Ể', 'ể', 'Ễ',
                 'ễ', 'Ệ', 'ệ', 'Ỉ', 'ỉ', 'Ị', 'ị', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ',
                 'ổ', 'Ỗ', 'ỗ', 'Ộ', 'ộ', 'Ớ', 'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'Ỡ', 'ỡ', 'Ợ', 'ợ', 'Ụ',
                 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'Ử', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ', 'Ỵ',
                 'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ', '–', '“', '”', '…', '<nul>', '  ']

(img_h_ADDRESS, img_w_ADDRESS, max_len_ADDRESS) = (40, 160, 12)
path_model_ADDRESS = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/ADDRESS.h5"
model_ADDRESS = Model(vocab_ADDRESS, img_w_ADDRESS, img_h_ADDRESS, max_len_ADDRESS, path_model_ADDRESS)
path = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped/infor_cropped/DATE_0406"
path_csv = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped/infor_cropped/DATE_0406.csv"

vocab_ID = "0123456789"
(img_h_ID, img_w_ID, max_len_ID) = (50, 250, 12)
path_model_ID = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/ID.h5"
model_ID = Model(vocab_ID, img_w_ID, img_h_ID, max_len_ID, path_model_ID)

vocab_NAME = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Á', 'Â', 'Ã', 'È',
                  'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'Ă', 'Đ', 'Ĩ', 'Ũ', 'Ơ',
                  'Ư', 'Ạ', 'Ả', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ẹ', 'Ẻ', 'Ẽ',
                  'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ỉ', 'Ị', 'Ọ', 'Ỏ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ',
                  'Ở', 'Ỡ', 'Ợ', 'Ụ', 'Ủ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Ỳ', 'Ỵ', 'Ỷ', 'Ỹ', '<nul>']
(img_h_NAME, img_w_NAME, max_len_NAME) = (50, 250, 6)
path_model_NAME = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/NAME.h5"
model_NAME = Model(vocab_NAME, img_w_NAME, img_h_NAME, max_len_NAME, path_model_NAME)

def to_csv_day():
    df = pandas.read_csv(path_csv)
    list_label = df['label']
    st = time.time()
    for file_img in glob.glob(f"{path}/*"):
        name = os.path.basename(file_img)
        index_name = df[df['filename'] == f'{name}'].index.values[0]
        img = cv2.imread(file_img)
        infor = model_BIRTHDAY.run(img, check=True)
        if list_label[index_name] != infor:
            print(f"Name: {name}| Label: {list_label}| Predict: {infor}")
    print(time.time() - st)


# path_add = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/DC_TG_GT_DT/ADDRESS"
# path_add_out = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/DC_TG_GT_DT/ADDRESS_WRONG"
# path_add_csv = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/DC_TG_GT_DT/ADDRESS.csv"
# path_add_out_csv = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/DC_TG_GT_DT/file_add.csv"
path_img_id = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/ID"
path_img_id_csv = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/ID.csv"

path_img_name = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/NAME"
path_img_name_csv = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/NAME.csv"
def run_model(model, path_img, path_csv):
    path_name_out = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/recoginition/NAME_WRONG"
    df = pandas.read_csv(path_csv, dtype={"label": str})
    label = df['label']
    i = 0
    for f in glob.glob(f"{path_img}/*"):
        img = cv2.imread(f)
        name = os.path.basename(f)
        infor = model_NAME.run(img)
        if  len(df[df['filename'] == f"{name}"].index.values) != 0:
            index = df[df['filename'] == f"{name}"].index.values[0]
        if f"{label[index]}" != f"{infor}":
            i += 1
            shutil.move(f"{path_img}/{name}", f"{path_name_out}/{name}")
            print(f"Name: {name}___{i}, Label: {label[index]}, Predict: {infor}")
run_model(model_NAME, path_img_name, path_img_name_csv)

def move_file_add_wrong():
    df = pandas.read_csv(path_add_csv)
    label = df["label"]
    df_new = pandas.read_csv(path_add_out_csv, header=0)
    print(df_new.columns)
    label_new = df_new['label']
    i = 0
    for _file in glob.glob(f"{path_add_out}/*"):
        name = os.path.basename(_file)
        try:
            index_name_new = df_new[df_new['filename'] == f'{name}'].index.values[0]
            index_name = df[df['filename']== f'{name}'].index.values[0]
            label_new[index_name_new] = label[index_name]
            print(label_new[index_name_new])
            # img = cv2.imread(_file)
            # infor = model_ADDRESS.run(img)
            # if label[index_name] != infor:
            #     i += 1
            #     print(f'{label[index_name]} | {infor}')
            #     shutil.move(f"{path_add}/{name}", f"{path_add_out}/{name}")
        except:
            i += 1
            print(name)
            # raise
    df_new.to_csv(path_add_out_csv, index=False, quoting=csv.QUOTE_ALL)
    print(i)
# move_file_add_wrong()
# df.to_csv(path_csv, index=False)
