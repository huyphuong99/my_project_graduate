import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from numpy import expand_dims
import glob
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import pandas as pd

path = "/home/huyphuong99/PycharmProjects/project_graduate/data/ID"


def add_padding(path_i):
    img = cv.imread(path_i)
    # name = os.path.basename(img_path)
    hh, ww, cc = img.shape
    try:
        rate = ww / hh
        img = cv.resize(img, (round(rate * img_h), img_h), interpolation=cv.INTER_AREA)
        color = (0, 0, 0)
        result = np.full((img_h, img_w, cc), color, dtype=np.uint8)
        result[:img_h, :round(rate * img_h)] = img
    except:
        img = cv.resize(img, (img_w, img_h))
    return img


def load_img(path_i):
    img_padd = add_padding(path_i)
    img = cv.cvtColor(img_padd, cv.COLOR_RGB2GRAY)
    img = img / 255.0
    img = cv.resize(img, (img_w, img_h))
    img = cv.transpose(img)
    img = img.reshape((1, img_w, img_h))
    return expand_dims(img, axis=-1)


def decode_img(predict):
    char_to_num = layers.experimental.preprocessing.StringLookup(
        vocabulary=list(characters), num_oov_indices=1, mask_token=''
    )
    num_to_char = layers.experimental.preprocessing.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), invert=True
    )
    input_len = np.ones(predict.shape[0]) * predict.shape[1]
    result = keras.backend.ctc_decode(predict, input_length=input_len, greedy=True)[0][0][:,
             :max_len]
    result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    return result.replace("[UNK]", "")


def acc(orig_text, predict, name):
    count = 0
    for i in range(len(orig_text)):
        if orig_text[i] == predict[i]:
            count += 1
        else:
            print(f"{name[i]} | {orig_text[i]} | {predict[i]}")
    acc = count / len(orig_text)
    print(f"Accuracy test data: {round(acc, 2) * 100}% / Total image: {count}")


def run():
    img_path = [f for f in glob.glob(f"{path}/*")]
    label = []
    name_img = []
    result = []
    _start = time.time()
    for i in img_path:
        name = os.path.basename(i)
        temp = list(df['filename'].values)
        if name in temp:
            lb = df["label"][temp.index(name)]
            label.append(lb)
        else:
            pass
        img_pre = load_img(i)
        predict = model.predict(img_pre)
        predict_text = decode_img(predict)
        name_img.append(name)
        result.append(predict_text)
        break
    _end = time.time()
    acc(label, result, name_img)
    print(f"Time is: {_end - _start} second")


if __name__ == "__main__":
    st = time.time()
    model = keras.models.load_model("weights/ID.h5")
    ed = time.time()
    print(f"Time load model is :{ed-st}")
    (img_h, img_w, max_len) = (50, 250, 12)
    characters = "0123456789"
    characters = sorted(characters)
    # (img_h, img_w, max_len) = (40, 210, 10)
    # characters = "-./1234567890"
    # characters.append("<nul>")
    # characters.append(" ")
    df = pd.read_csv(f"{path}.csv", dtype={"label": str})
    run()
