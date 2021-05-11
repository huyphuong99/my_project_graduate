import random
import os
import csv
import cv2
import numpy as np

path = "/home/huyphuong99/PycharmProjects/project_graduate/data/NAME/"
file_ = os.listdir(path)
output_path = "/home/huyphuong99/PycharmProjects/project_graduate/data/FULL_NAME/"
with open("/home/huyphuong99/PycharmProjects/project_graduate/data/FULL_NAME.csv", "w",
          newline="") as _file:
    writer = csv.writer(_file)
    writer.writerow(["filename", "label"])
    for idx in range(5000):
        try:
            name_img1 = random.choice(file_)
            name_img2 = random.choice(file_)
            name_img3 = random.choice(file_)
            name_img4 = random.choice(file_)
            name1 = name_img1.split("_")[-1].replace(".png", "")
            name2 = name_img2.split("_")[-1].replace(".png", "")
            name3 = name_img3.split("_")[-1].replace(".png", "")
            name4 = name_img4.split("_")[-1].replace(".png", "")
            # Read img
            img1 = cv2.imread(path + name_img1, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(path + name_img2, cv2.IMREAD_GRAYSCALE)
            img3 = cv2.imread(path + name_img3, cv2.IMREAD_GRAYSCALE)
            img4 = cv2.imread(path + name_img4, cv2.IMREAD_GRAYSCALE)
            # Take shape img
            h1, w1 = img1.shape
            rate1 = w1 / h1
            h2, w2 = img2.shape
            rate2 = w2 / h2
            h3, w3 = img3.shape
            rate3 = w3 / h3
            h4, w4 = img4.shape
            rate4 = w4 / h4
            h_mean = int((h1 + h2 + h3 + h4) / 4)
            img1 = cv2.resize(img1, (int(rate1 * h_mean), h_mean), interpolation=cv2.INTER_AREA)
            img1 = np.delete(img1, np.where(~img1.any(axis=0))[0], axis=1)
            img2 = cv2.resize(img2, (int(rate2 * h_mean), h_mean), interpolation=cv2.INTER_AREA)
            img2 = np.delete(img2, np.where(~img2.any(axis=0))[0], axis=1)
            img3 = cv2.resize(img3, (int(rate3 * h_mean), h_mean), interpolation=cv2.INTER_AREA)
            img3 = np.delete(img3, np.where(~img3.any(axis=0))[0], axis=1)

            img4 = cv2.resize(img4, (int(rate4 * h_mean), h_mean), interpolation=cv2.INTER_AREA)
            img4 = np.delete(img4, np.where(~img4.any(axis=0))[0], axis=1)

            pad_1 = np.array([img1[:, -1].tolist()] * int(w1 * 0.1 / 2)).astype(np.uint8).T
            pad_2 = np.array([img2[:, 0].tolist()] * int(w2 * 0.1 / 2)).astype(np.uint8).T
            pad_3 = np.array([img2[:, -1].tolist()] * int(w2 * 0.1 / 2)).astype(np.uint8).T
            pad_4 = np.array([img3[:, 0].tolist()] * int(w3 * 0.1 / 2)).astype(np.uint8).T
            pad_5 = np.array([img3[:, -1].tolist()] * int(w3 * 0.1 / 2)).astype(np.uint8).T
            pad_6 = np.array([img4[:, 0].tolist()] * int(w4 * 0.1 / 2)).astype(np.uint8).T
            result = cv2.hconcat([img1, pad_1, pad_2, img2])
            print(f"File name is: {name1}_{name2}")
            cv2.imwrite(f"{output_path}{name1}_{name2}.jpg", result)
            writer.writerow([f"{name1}_{name2}.jpg", f"{name1} {name2}"])
        except:
            pass

    print("Done!!!")
