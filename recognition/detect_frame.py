import numpy as np
import requests
import ast
import base64
import os
import cv2
from matplotlib._layoutbox import seq_id

from model_ocr import Model
import time

url = "http://thienntpc.local:15004/image/ocr/id/label"
file_img = "/home/huyphuong99/PycharmProjects/project_graduate/temp_compress/temp/0000787878787808.jpg"
file_ = {"image": open(file_img, "rb")}
res = requests.post(url, files=file_)
res = res.content.decode("UTF-8")
res = ast.literal_eval(res)
side_img = res['result'][0]["side"]
dict_img = res['result'][0]['info']
img_box = res['result'][0]


def inf_field(key, value):
    key = key.upper()
    field_i = []
    name_i = []
    for j in range(len(value)):
        img_ = base64.b64decode(value[j]["image"])
        name = key + "_" + str(j) + "_" + value[j]["label"]
        img_ = np.frombuffer(img_, dtype=np.uint8)
        img = cv2.imdecode(img_, flags=cv2.IMREAD_COLOR)
        field_i.append(img)
        name_i.append(name)
    return field_i, name_i


def return_img():
    dict_img_ = {}
    for key, value in dict_img.items():
        field_i, label = inf_field(key, value)
        dict_img_[key.upper()] = field_i
    return dict_img_


if __name__ == "__main__":
    vocab_fullname = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                      'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                      'e', 'f', 'g', 'h', 'i',
                      'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                      'y', 'z', "<nul>", " "]

    vocab_ID = "0123456789"
    (img_h_ID, img_w_ID, max_len_ID) = (50, 250, 12)
    path_model_ID = "/home/huyphuong99/PycharmProjects/project_graduate/weights/ID.h5"
    vocab_NAME = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Á', 'Â', 'Ã', 'È',
                  'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'Ă', 'Đ', 'Ĩ', 'Ũ', 'Ơ',
                  'Ư', 'Ạ', 'Ả', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ẹ', 'Ẻ', 'Ẽ',
                  'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ỉ', 'Ị', 'Ọ', 'Ỏ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ',
                  'Ở', 'Ỡ', 'Ợ', 'Ụ', 'Ủ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Ỳ', 'Ỵ', 'Ỷ', 'Ỹ', '<nul>']
    (img_h_NAME, img_w_NAME, max_len_NAME) = (50, 250, 6)
    path_model_NAME = "/home/huyphuong99/PycharmProjects/project_graduate/weights/NAME.h5"

    vocab_DAY = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<nul>', ' ']
    (img_h_DAY, img_w_DAY, max_len_DAY) = (40, 210, 12)
    path_model_DAY = "/home/huyphuong99/PycharmProjects/project_graduate/weights/BIRTHDAY.h5"

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
    path_model_ADDRESS = "/home/huyphuong99/PycharmProjects/project_graduate/weights/ADDRESS.h5"
    model_ID = Model(vocab_ID, img_w_ID, img_h_ID, max_len_ID, path_model_ID)
    model_NAME = Model(vocab_NAME, img_w_NAME, img_h_NAME, max_len_NAME, path_model_NAME)
    model_DAY = Model(vocab_DAY, img_w_DAY, img_h_DAY, max_len_DAY, path_model_DAY)
    model_ADDRESS = Model(vocab_ADDRESS, img_w_ADDRESS, img_h_ADDRESS,
                          max_len_ADDRESS, path_model_ADDRESS)
    st = time.time()
    dict_image = return_img()
    dict_ = {}
    for key, value in dict_image.items():
        temp = ""
        print(key)
        for img in value:
            global t
            if key == "ADDRESS" or key == "ISSUE_BY" or key == "GENDER" or key == "RELIGION" or key == "ETHNICITY":
                t = model_ADDRESS.run(img)
            elif key == "NAME":
                t = model_NAME.run(img)
            elif key == "BIRTHDAY" or key == "EXPIRY" or key == "ISSUE_DATE":
                t = model_DAY.run(img)
            else:
                t = model_ID.run(img)
            temp += f"{t} "
        dict_[key] = temp

    print("Thông tin chứng minh nhân dân:")
    print("=================================================================")
    if side_img == "FRONT":
        print(f"Họ và tên: {dict_['NAME']}")
        print(f"Số CMND: {dict_['ID']}")
        print(f"Sinh ngày: {dict_['BIRTHDAY']}")
        print(f"Nguyên quán / Nơi đăng ký HKTT: {dict_['ADDRESS']}")
    else:
        print(f"Dân tộc: {dict_['RELIGION']}")
        print(f"Tôn giáo: {dict_['ETHNICITY']}")
        print(f"Ngày cấp: {dict_['ISSUE_DATE'].replace(' ', '-')[:-1]}")
        print(f"Nơi cấp: {dict_['ISSUE_BY']}")
    print("=================================================================")
    print(f"{time.time() - st}")
