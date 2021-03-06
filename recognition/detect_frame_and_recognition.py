import time
import tensorflow as tf
import os
from file_detection.load_model_detect_frame import main_detection
from recognition.model_ocr import Model
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import config_util

# CONFIG MODEL DETECTION
PATH_TO_LABEL_FRAME = "/home/huyphuong99/PycharmProjects/project_graduate/file_model_detect_frame/label_map.pbtxt"
PATH_TO_MODEL_DIR_FRAME = "/home/huyphuong99/PycharmProjects/project_graduate/file_model_detect_frame"
PATH_TO_CONFIG_FRAME = PATH_TO_MODEL_DIR_FRAME + "/pipeline.config"
PATH_TO_CKPT_FRAME = PATH_TO_MODEL_DIR_FRAME + "/checkpoint"
NAME_MODEL_FRAME = 'ckpt-21'
PATH_TO_LABEL_INFOR = "/home/huyphuong99/PycharmProjects/project_graduate/file_model_detect_infor/label_map.pbtxt"
PATH_TO_MODEL_INFOR = "/home/huyphuong99/PycharmProjects/project_graduate/file_model_detect_infor"
PATH_TO_CONFIG_INFOR = PATH_TO_MODEL_INFOR + "/pipeline.config"
PATH_TO_CKPT_INFOR = PATH_TO_MODEL_INFOR + "/checkpoint"
NAME_MODEL_INFOR = "ckpt-45"

# CONFIG DETECTION FRAME
configs_fr = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG_FRAME)
model_config_fr = configs_fr['model']
detection_model_fr = model_builder.build(model_config=model_config_fr, is_training=False)
ckpt_fr = tf.compat.v2.train.Checkpoint(model=detection_model_fr)
ckpt_fr.restore(os.path.join(PATH_TO_CKPT_FRAME, NAME_MODEL_FRAME)).expect_partial()
category_index_frame = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_FRAME,
                                                                          use_display_name=True)

# CONFIG DETECITON INFOR
configs_inf = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG_INFOR)
model_config_inf = configs_inf['model']
detection_model_inf = model_builder.build(model_config=model_config_inf, is_training=False)
ckpt_inf = tf.compat.v2.train.Checkpoint(model=detection_model_inf)
ckpt_inf.restore(os.path.join(PATH_TO_CKPT_INFOR, NAME_MODEL_INFOR)).expect_partial()
category_index_infor = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_INFOR,
                                                                          use_display_name=True)

# CONFIG MODEL RECOGNITION
vocab_fullname = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
                  'e', 'f', 'g', 'h', 'i',
                  'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z', "<nul>", " "]

vocab_ID = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(img_h_ID, img_w_ID, max_len_ID) = (50, 250, 12)
path_model_ID = "/home/huyphuong99/PycharmProjects/project_graduate/recognition/weights/ID_ver1_95.45%.h5"

vocab_NAME = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
              'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '??', '??', '??', '??', '??',
              '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??',
              '??', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
              '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
              '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '<nul>']
(img_h_NAME, img_w_NAME, max_len_NAME) = (50, 250, 6)
path_model_NAME = "/home/huyphuong99/PycharmProjects/project_graduate/recognition/weights/NAME.h5"

vocab_DAY = ['-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
(img_h_DAY, img_w_DAY, max_len_DAY) = (40, 210, 10)
path_model_BIRTHDAY = "/home/huyphuong99/PycharmProjects/project_graduate/recognition/weights/DATE_010621_ver3_final.h5"

# vocab_ADDRESS = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0',
#                  '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
#                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
#                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
#                  '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
#                  'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '??',
#                  '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??',
#                  '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??',
#                  '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '???', '???', '???', '???', '???',
#                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
#                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
#                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
#                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
#                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???',
#                  '???', '???', '???', '???', '???', '???', '???', '???', '???', '<nul>', '  ']
vocab_ADDRESS = ['!', '"', '#', '$', '%', '&', "'", "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '??', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???', '???']
(img_h_ADDRESS, img_w_ADDRESS, max_len_ADDRESS) = (40, 160, 12)
path_model_ADDRESS = "/home/huyphuong99/PycharmProjects/project_graduate/recognition/weights/ADDRESS_210621_95.73%.h5"

model_ID = Model(vocab_ID, img_w_ID, img_h_ID, max_len_ID, path_model_ID)
model_NAME = Model(vocab_NAME, img_w_NAME, img_h_NAME, max_len_NAME, path_model_NAME)
model_BIRTHDAY = Model(vocab_DAY, img_w_DAY, img_h_DAY, max_len_DAY, path_model_BIRTHDAY)
model_ADDRESS = Model(vocab_ADDRESS, img_w_ADDRESS, img_h_ADDRESS,
                      max_len_ADDRESS, path_model_ADDRESS)


def assign_model(dict_image, model_ID, model_NAME, model_BIRTHDAY, model_ADDRESS):
    dict_ = {}
    for key, value in dict_image.items():
        t = None
        if key == "ADDRESS" or key == "DOMICILE" or key == "COUNTRY" or key == "SEX" or key == "ISSUE BY" or \
                key == "GENDER" or key == "RELIGION" or key == "ETHNICITY":
            t = model_ADDRESS.run(value, check=True)
        elif key == "NAME":
            t = model_NAME.run(value)
        elif key == "BIRTHDAY" or key == "EXPIRY" or key == "ISSUE DATE":
            t = model_BIRTHDAY.run(value, check=True)
            if len(t) < 5:
                t = model_BIRTHDAY.run(value, check=True)
        elif key == "ID":
            t = model_ID.run(value, check=True)
        dict_[key] = t
    return dict_


def print_infor(dict_, file_name):
    print("=================================================================")
    print("Th??ng tin ch???ng minh nh??n d??n:")
    print(f"???nh: {file_name}")
    if "NAME" in dict_.keys():
        print(f"S???: {dict_['ID']}")
        print(f"H??? v?? t??n: {dict_['NAME']}")
        if "SEX" in dict_.keys():
            print(f"Gi???i t??nh: {dict_['SEX']}")
            if "COUNTRY" in dict_.keys():
                print(f"Qu???c t???ch: {dict_['COUNTRY']}")
            elif "ETHNICITY" in dict_.keys():
                print(f"D??n t???c: {dict_['ETHNICITY']}")
        if "EXPIRY" in dict_.keys():
            print(f"C?? gi?? tr??? ?????n: {dict_['EXPIRY']}")
        if "BIRTHDAY" in dict_.keys():
            print(f"Sinh ng??y: {dict_['BIRTHDAY']}")
        print(f"Nguy??n qu??n: {dict_['DOMICILE']}")
        print(f"N??i ????ng k?? HKTT: {dict_['ADDRESS']}")
    else:
        if "ETHNICITY" in dict_.keys():
            print(f"D??n t???c: {dict_['ETHNICITY']}")
            print(f"T??n gi??o: {dict_['RELIGION']}")
        print(f"Ng??y c???p: {dict_['ISSUE DATE']}")
        print(f"N??i c???p: {dict_['ISSUE BY']}")
    print("=================================================================")


def main_project(path_api, ratio):
    path = [path_api]
    for i in range(len(path)):
        try:
            file_name = os.path.basename(path[i])
            dict_image = main_detection(path[i], detection_model_fr, category_index_frame, detection_model_inf,
                                        category_index_infor, file_name, ratio)
            # dict_ = assign_model(dict_image, model_ID, model_NAME, model_BIRTHDAY, model_ADDRESS)
            # print_infor(dict_, file_name)
            # return dict_
        except Exception as e:
            print(e)
            raise e


if __name__ == "__main__":
    st = time.time()
    path = "/home/huyphuong99/Desktop/material/test/id/cccd_new_3.jpg"
    main_project(path)
    print(time.time()-st)
