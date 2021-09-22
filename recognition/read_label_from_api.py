import glob
import base64
import os
import cv2
import requests
import ast
import  time
path_img = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_image/raw_image/*"
out_path_front = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_image/cropped_raw_image/"
out_path_back = "/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/ouput/back/"
out_card_front = "/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/ouput/card/front/"
out_card_back = "/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/ouput/card/back/"
url = "http://172.16.1.11:15004/image/ocr/id/label"
id_card = "/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/ouput/card/"


def inf_feild(output_path, key, value):
    for j in range(len(value)):
        img_ = base64.b64decode(value[j]["image"])
        name = value[j]["label"]
        fl = os.path.basename(file_img).replace(".jpg", "")
        path = f"{output_path}{key}/{fl}_{j}_{name}.jpg"
        if os.path.exists(path):
            break
        with open(path, 'wb') as f:
            f.write(img_)

def img_(output_path, key, value):
    img_ = base64.b64decode(value)
    with open(output_path+key, "wb") as f:
        f.write(img_)


i = 0
lst_id = [os.path.basename(file).split("_")[0] for file in glob.glob(
    "/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/ouput/front/id/*")]
lst_issue_date = [os.path.basename(file).split("_")[0] for file in glob.glob(
    "/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/ouput/back/issue_date/*")]

for file_img in glob.glob(path_img):
    _start = time.time()
    i += 1
    chk_front = os.path.basename(file_img).replace(".jpg", "")
    chk_back = os.path.basename(file_img).replace(".jpg", "")

    # if chk_front in lst_id or chk_back in lst_issue_date:
    #     print(f"File {file_img} existed!!!")
    #     continue

    file_ = {"image": open(file_img, "rb")}
    res = requests.post(url, files=file_)
    res = res.content.decode("UTF-8")
    res = ast.literal_eval(res)
    print(f"File number {i}")
    try:
        side_img = res['result'][0]["side"]
        dict_img = res['result'][0]['info']
        img_box = res['result'][0]
        if side_img == "FRONT":
            for key, value in dict_img.items():
                inf_feild(out_path_front, key, value)
            for key, value in img_box:
                img_(out_card_front, key, value)

        elif side_img == "BACK":
            for key, value in dict_img.items():
                inf_feild(out_path_back, key, value)
            for key, value in img_box:
                img_(out_card_back, key, value)


    except:
        print(file_img)
        os.remove(file_img)
    _end = time.time()
    print(f"Time loop is: {_end - _start}s")
