from tensorflow import keras
import cv2
import numpy as np
from numpy import expand_dims
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import time


class Model():
    def __init__(self, vocab, img_w, img_h, max_len, model):
        # self.path = path
        self.vocab = vocab
        self.img_w = img_w
        self.img_h = img_h
        self.max_len = max_len
        self.model = keras.models.load_model(model)

    def add_padding(self, img):
        # img = cv2.imread(img)
        img = img
        hh, ww, cc = img.shape
        try:
            rate = ww / hh
            img = cv2.resize(img, (round(rate * self.img_h), self.img_h),
                             interpolation=cv2.INTER_AREA)
            color = (0, 0, 0)
            result = np.full((self.img_h, self.img_w, cc), color, dtype=np.uint8)
            result[:self.img_h, :round(rate * self.img_h)] = img
        except:
            result = cv2.resize(img, (self.img_w, self.img_h))
        return result

    def load_img(self, img):
        img = self.add_padding(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        img = img / 255.0
        # img = cv2.resize(img, (self.img_w, self.img_h))
        img = cv2.transpose(img)
        img = img.reshape((1, self.img_w, self.img_h))
        return expand_dims(img, axis=-1)

    def decode_img(self, predict, check):
        char2idx = {}
        idx2char = {}
        for i, c in enumerate(self.vocab):
            char2idx[c] = i
            idx2char[i] = c
        def decode_text(indices):
            chars = None
            if check:
                chars = [idx2char[i] if (i != -1) else '' for i in indices]
            elif not check:
                chars = [idx2char[i-2] if (i != -1) else '' for i in indices]
            return "".join(chars)
        input_len = np.ones(predict.shape[0]) * predict.shape[1]
        result = keras.backend.ctc_decode(predict, input_length=input_len, greedy=True)[0][0][:,
                 :self.max_len]
        result = decode_text(list(result.numpy()[0]))
        # print(result)
        return result

    # def decode_img_(self, predict):
    #     char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(self.vocab),
    #                                                                  num_oov_indices=1,
    #                                                                  mask_token="")
    #     num_to_char = layers.experimental.preprocessing.StringLookup(
    #         vocabulary=char_to_num.get_vocabulary(), invert=True)
    #     input_len = np.ones(predict.shape[0]) * predict.shape[1]
    #     result = keras.backend.ctc_decode(predict, input_length=input_len, greedy=True)[0][0][:,
    #              :self.max_len]
    #     print(result)
    #     result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    #     print(result)
    #     return result.replace("[UNK]", "")

    def run(self, img, check=False):
        # name = os.path.basename(self.path)
        img = self.load_img(img)
        img_predict = self.model.predict(img)
        predict_text = self.decode_img(img_predict, check)
        return  predict_text
        # print(f" ---> {predict_text} <---")


if __name__ == "__main__":
    # img_ID = "00000bffffc80000_0_385008802.jpg"
    # path_ID = f"/home/huyphuong99/PycharmProjects/project_graduate/data/ID/{img_ID}"
    # vocab_ID = "0123456789"
    # (img_h_ID, img_w_ID, max_len_ID) = (50, 250, 12)
    # path_model_ID = "/home/huyphuong99/PycharmProjects/project_graduate/weights/ID.h5"
    img_NAME = "a1.jpg"
    path_NAME = f"/home/huyphuong/Desktop/material/test/{img_NAME}"
    vocab_NAME = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                  'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'À', 'Á', 'Â', 'Ã', 'È',
                  'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'Ă', 'Đ', 'Ĩ', 'Ũ', 'Ơ',
                  'Ư', 'Ạ', 'Ả', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ẹ', 'Ẻ', 'Ẽ',
                  'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ', 'Ỉ', 'Ị', 'Ọ', 'Ỏ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ',
                  'Ở', 'Ỡ', 'Ợ', 'Ụ', 'Ủ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự', 'Ỳ', 'Ỵ', 'Ỷ', 'Ỹ', '<nul>']
    (img_h_NAME, img_w_NAME, max_len_NAME) = (50, 500, 30)
    path_model_NAME = "/home/huyphuong/PycharmProjects/project_graduate/recognition/weights/FULL_NAME.h5"
    #
    # img_DAY = "ISSUEDATE/2_0_26.png"
    # path_DAY = f"/home/huyphuong99/PycharmProjects/project_graduate/data/DAY/{img_DAY}"
    # vocab_DAY = ['-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<nul>', ' ']
    # (img_h_DAY, img_w_DAY, max_len_DAY) = (40, 210, 12)
    # path_model_DAY = "/home/huyphuong99/PycharmProjects/project_graduate/weights/BIRTHDAY.h5"
    #
    # img_ADDRESS = "DC_TG_GT_DT/ADDRESS/2_2_TT.png"
    # path_ADDRESS = f"/home/huyphuong99/PycharmProjects/project_graduate/data/{img_ADDRESS}"
    # vocab_ADDRESS = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0',
    #                  '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@',
    #                  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    #                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    #                  '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    #                  'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'À',
    #                  'Á', 'Â', 'Ã', 'È', 'É', 'Ê', 'Ì', 'Í', 'Ò', 'Ó', 'Ô', 'Õ', 'Ù', 'Ú', 'Ý', 'à',
    #                  'á', 'â', 'ã', 'è', 'é', 'ê', 'ì', 'í', 'ò', 'ó', 'ô', 'õ', 'ù', 'ú', 'ý', 'Ă',
    #                  'ă', 'Đ', 'đ', 'Ĩ', 'ĩ', 'Ũ', 'ũ', 'Ơ', 'ơ', 'Ư', 'ư', 'Ạ', 'ạ', 'Ả', 'ả', 'Ấ',
    #                  'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ', 'Ậ', 'ậ', 'Ắ', 'ắ', 'Ằ', 'ằ', 'Ẳ', 'ẳ', 'Ẵ',
    #                  'ẵ', 'Ặ', 'ặ', 'Ẹ', 'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ', 'Ế', 'ế', 'Ề', 'ề', 'Ể', 'ể', 'Ễ',
    #                  'ễ', 'Ệ', 'ệ', 'Ỉ', 'ỉ', 'Ị', 'ị', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ',
    #                  'ổ', 'Ỗ', 'ỗ', 'Ộ', 'ộ', 'Ớ', 'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'Ỡ', 'ỡ', 'Ợ', 'ợ', 'Ụ',
    #                  'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'Ử', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ', 'Ỵ',
    #                  'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ', '–', '“', '”', '…', '<nul>', '  ']
    #
    # (img_h_ADDRESS, img_w_ADDRESS, max_len_ADDRESS) = (40, 160, 12)
    # path_model_ADDRESS = "/home/huyphuong99/PycharmProjects/project_graduate/weights/ADDRESS.h5"

    # model_ID = Model(path_ID, vocab_ID, img_w_ID, img_h_ID, max_len_ID, path_model_ID)
    model_NAME = Model(vocab_NAME, img_w_NAME, img_h_NAME, max_len_NAME, path_model_NAME)
    # model_DAY = Model(path_DAY, vocab_DAY, img_w_DAY, img_h_DAY, max_len_DAY, path_model_DAY)
    # model_ADDRESS = Model(path_ADDRESS, vocab_ADDRESS, img_w_ADDRESS, img_h_ADDRESS, max_len_ADDRESS, path_model_ADDRESS)

    # model_ID.run()
    # predict = model_NAME.run(path_NAME)
    # print(predict)
    # # model_ADDRESS.run()
    # model_DAY.run()
