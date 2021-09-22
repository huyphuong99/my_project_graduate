import csv
import glob
import os
import statistics as st
from model_ocr import Model
import cv2
import pandas as pd

path = "/media/huyphuong/huyphuong99/tima/project/passport/raw_data/cropped/path_cropped_inf/issue_by"


vocab_NAME = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Á', 'Â', 'Ã', 'È',
              'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'Ă', 'Đ', 'Ĩ', 'Ũ', 'Ơ',
              'Ư', 'Ạ', 'Ả', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ẹ', 'Ẻ', 'Ẽ',
              'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ỉ', 'Ị', 'Ọ', 'Ỏ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ',
              'Ở', 'Ỡ', 'Ợ', 'Ụ', 'Ủ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Ỳ', 'Ỵ', 'Ỷ', 'Ỹ', '<nul>']
(img_h_NAME, img_w_NAME, max_len_NAME) = (50, 250, 6)
path_model_NAME = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/NAME.h5"
model_NAME = Model(vocab_NAME, img_w_NAME, img_h_NAME, max_len_NAME, path_model_NAME)


vocab_ADDRESS = ['!', '"', '#', '$', '%', '&', "'", "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'À', 'Á', 'Â', 'Ã', 'È', 'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'à', 'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'Ă', 'ă', 'Đ', 'đ', 'Ĩ', 'ĩ', 'Ũ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư', 'Ạ', 'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ', 'Ậ', 'ậ', 'Ắ', 'ắ', 'Ằ', 'ằ', 'Ẳ', 'ẳ', 'Ẵ', 'ẵ', 'Ặ', 'ặ', 'Ẹ', 'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ', 'Ế', 'ế', 'Ề', 'ề', 'Ể', 'ể', 'Ễ', 'ễ', 'Ệ', 'ệ', 'Ỉ', 'ỉ', 'Ị', 'ị', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ', 'ổ', 'Ỗ', 'ỗ', 'Ộ', 'ộ', 'Ớ', 'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'Ỡ', 'ỡ', 'Ợ', 'ợ', 'Ụ', 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'Ử', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ', 'Ỵ', 'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ', '–', '“', '”']
(img_h_ADDRESS, img_w_ADDRESS, max_len_ADDRESS) = (40, 160, 12)
path_model_ADDRESS = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/ADDRESS_210621_95.73%.h5"
model_ADDRESS = Model(vocab_ADDRESS, img_w_ADDRESS, img_h_ADDRESS,
                      max_len_ADDRESS, path_model_ADDRESS)

vocab_ID = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(img_h_ID, img_w_ID, max_len_ID) = (50, 250, 12)
path_model_ID = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/ID_ver1_95.45%.h5"
model_ID = Model(vocab_ID, img_w_ID, img_h_ID, max_len_ID, path_model_ID)

vocab_DAY = ['-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(img_h_DAY, img_w_DAY, max_len_DAY) = (40, 210, 10)
path_model_BIRTHDAY = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/DATE_010621_ver3_final.h5"
model_BIRTHDAY = Model(vocab_DAY, img_w_DAY, img_h_DAY, max_len_DAY, path_model_BIRTHDAY)

df = pd.read_csv(path + ".csv", dtype={"filename": str, "label": str})
# xxx = df.to_dict()
label = df["label"]
confidence = df["conf"]
for file in glob.glob(path + "/*"):
    name = os.path.basename(file)
    img = cv2.imread(file)
    lb, conf = model_ADDRESS.run([img], check=True)
    conf = st.mean(conf[0])
    index = df[df["filename"]==name].index.values[0]
    label[index] = lb
    confidence[index] = conf
    print(lb, conf)
#     xxx['label'][index] = x
# df = pd.DataFrame(xxx)
print(df)
df.to_csv(path +".csv", index=False, quoting=csv.QUOTE_ALL)
    # print(t)
#     if t1 == t:
#         count += 1
#     else:
#         print(f"{t}, {t1}")
#         a += 1
#         if t.count(" ") == t1.count(" "):
#             b += 1
# print(f"Acc: {round(count / total * 100, 2)}%")
# print(f"{1 - b / (a+0.000002)}")
