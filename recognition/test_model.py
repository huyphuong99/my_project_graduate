import glob
import os

from model_ocr import Model
import cv2
import pandas as pd

path = "/home/huyphuong99/PycharmProjects/project_graduate/temp_compress/output_"

characters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Á', 'Â', 'Ã', 'È',
              'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'Ă', 'Đ', 'Ĩ', 'Ũ', 'Ơ',
              'Ư', 'Ạ', 'Ả', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ẹ', 'Ẻ', 'Ẽ',
              'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ỉ', 'Ị', 'Ọ', 'Ỏ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ',
              'Ở', 'Ỡ', 'Ợ', 'Ụ', 'Ủ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Ỳ', 'Ỵ', 'Ỷ', 'Ỹ', '<nul>', "  "]
(img_h_ID, img_w_ID, max_len_ID) = (50, 500, 30)
path_model = "/home/huyphuong99/PycharmProjects/project_graduate/weights/output_.h5"
model = Model(characters, img_w_ID, img_h_ID, max_len_ID, path_model)

df = pd.read_csv(path + ".csv")
label = df["label"]
count = 0
total = 0
a = 0
b = 0
for file in glob.glob(path + "/*"):
    total += 1
    t1 = os.path.basename(file).split("_")[-1].replace(".jpg", "")
    img = cv2.imread(file)
    t = model.run(img)
    if t1 == t:
        count += 1
    else:
        print(f"{t}, {t1}")
        a += 1
        if t.count(" ") == t1.count(" "):
            b += 1
print(f"Acc: {round(count / total * 100, 2)}%")
print(f"{1 - b / (a+0.000002)}")

# import requests
# import ast
# import base64
# import os
#
# url_ = "http://thienntpc.local:15004/image/ocr/id/label"
# url = "http://thiennt.local:15004/image/ocr/id/label"
# file_img = "/home/huyphuong99/PycharmProjects/project_graduate/temp_compress/temp/*"
# output = "/home/huyphuong99/PycharmProjects/project_graduate/temp_compress/output_/"
#
#
# def img_(output_path, f, key, value):
#     img_ = base64.b64decode(value)
#     with open(output_path + f + "_" + key + ".jpg", "wb") as f:
#         f.write(img_)
#
#
# i = 0
# lst_id = [os.path.basename(file).split("_")[0] for file in
#           glob.glob("/home/huyphuong99/PycharmProjects/project_graduate/temp_compress/output_fn/*")]
#
# for f in glob.glob(file_img):
#     fl = os.path.basename(f).replace(".jpg", "")
#     i += 1
#     try:
#         file_ = {"image": open(f, "rb")}
#         res = requests.post(url, files=file_)
#         res = res.content.decode("UTF-8")
#         res = ast.literal_eval(res)
#         side_img = res['result'][0]["side"]
#         dict_img = res['result'][0]['info']
#         if side_img == "FRONT":
#             img_box = res['result'][0]['info_full']['name']
#             image_fn = img_box["image"]
#             label_fn = img_box["label"]
#             img_(output, fl, label_fn, image_fn)
#             print(f"Image is {i}")
#     except:
#         pass
