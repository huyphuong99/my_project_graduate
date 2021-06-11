import glob
import numpy as np
import random
import cv2
import json
import os


def rotation(img, box, coordinate):
    variance = 20
    angle1, angle2, angle3, angle4 = random.gauss(0, variance), random.gauss(90,
                                                                             variance), random.gauss(
        180, variance), random.gauss(270, variance)
    angle = random.choice([angle1, angle2, angle3, angle4])
    height, width, channels = img.shape
    center_x, center_y = width // 2, height // 2
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int(height * sin + width * cos)
    nH = int(height * cos + width * sin)
    M[0, 2] += (nW / 2) - center_x
    M[1, 2] += (nH / 2) - center_y
    img = cv2.warpAffine(img, M, (nW, nH))
    corners_new = np.hstack((coordinate[0, 0].reshape(-1, 1), coordinate[0, 1].reshape(-1, 1),
                            coordinate[1, 0].reshape(-1, 1), coordinate[1, 1].reshape(-1, 1),
                            coordinate[2, 0].reshape(-1, 1), coordinate[2, 1].reshape(-1, 1),
                            coordinate[3, 0].reshape(-1, 1), coordinate[3, 1].reshape(-1, 1)))
    corners = corners_new.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))
    corners = np.dot(M, corners.T).T
    corners = corners.reshape(-1, 8)
    return img, corners


def read_json(path_json):
    with open(path_json) as path:
        data = json.load(path)
        box = []
        label = None
        img_path = data["imagePath"]
        version = data["version"]
        for member in data["shapes"]:
            global coordinate
            coordinate = member["points"]
            label = member["label"]
            x = [x[0] for x in coordinate]
            y = [y[1] for y in coordinate]
            x_min, x_max, y_min, y_max = int(min(x)), int(max(x)), int(min(y)), int(max(y))
            box.append([x_min, y_min, x_max, y_max])
        return np.array(box), label, img_path, version, np.array(coordinate)


def write_json_file(path_save, name, box_, label, img_path, height, width, version):
    dict_ = {}
    dict_["version"] = version
    dict_["flags"] = {}
    shape = []
    d_in_shape = {}
    d_in_shape["label"] = label
    d_in_shape["points"] = box_
    d_in_shape["group_id"] = None
    d_in_shape["shape_type"] = "polygon"
    d_in_shape["flags"] = {}
    shape.append(d_in_shape)

    dict_["shapes"] = shape
    dict_["imagePath"] = f"aug_{img_path}"
    dict_["imageData"] = None
    dict_["imageHeight"] = height
    dict_["imageWidth"] = width
    json_object = json.dumps(dict_)

    with open(f"{path_save}aug_{name}", "w") as outfile:
        outfile.write(json_object)



if __name__ == "__main__":
    path = "/home/huyphuong99/PycharmProjects/project_graduate/id_data_raw_label/all_id/*.jpg"
    path_save = "/home/huyphuong99/PycharmProjects/project_graduate/id_data_raw_label/all_id_augment/"
    i = 0
    for path_img in glob.glob(path):
        try:
            img_name = os.path.basename(path_img)
            i += 1
            print(f"File number {i}:{img_name}")
            path_json = path_img.replace("jpg", "json")
            img = cv2.imread(path_img)
            box, label, img_path, version, coordinate = read_json(path_json)
            img_new, pts = rotation(img, box, coordinate)
            cv2.imwrite(path_save +"aug_"+ img_name, img_new)
            
            pts = np.array(pts)
            box_ = pts.reshape(-1, 2).tolist()
            pts = pts.reshape((-1, 1, 2))
            height, width, cc = img_new.shape
            write_json_file(path_save, os.path.basename(path_json), box_, label, img_path, height,
                            width, version)
            img_ = cv2.polylines(img_new, np.int32([pts]), True, (255, 0, 0), 2)
        except Exception as e:
            print(e)
