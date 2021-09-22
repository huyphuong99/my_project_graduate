import glob
import numpy as np
import os.path
import cv2
from enum import Enum


class Corner(Enum):
    # order = 'TOP_LEFT TOP_RIGHT BOTTOM_RIGHT BOTTOM_LEFT'
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3


def load_model_detect_frame(path, path_output, detection_model, category_index) -> object:
    IMAGE_PATHS_FRAME = path
    filename = os.path.basename(path)
    try:
        IMAGE_CROPPED_FRAME, label_frame, boxes_frame = run(IMAGE_PATHS_FRAME, detection_model, category_index)
        image = IMAGE_CROPPED_FRAME[0]
        cv2.imwrite(f"{path_output}/cr_{filename}", image)
    except:
        print(filename)


def list_to_dict(keypoint: list, box: list, keypoint_score: list):
    dict_keypoint = {}
    dict_box = {}
    dict_keypoint[Corner.TOP_LEFT] = {'coord': keypoint[0], 'score': keypoint_score[0]}
    dict_keypoint[Corner.TOP_RIGHT] = {'coord': keypoint[1], 'score': keypoint_score[1]}
    dict_keypoint[Corner.BOTTOM_RIGHT] = {'coord': keypoint[2], 'score': keypoint_score[2]}
    dict_keypoint[Corner.BOTTOM_LEFT] = {'coord': keypoint[3], 'score': keypoint_score[3]}
    dict_box[Corner.TOP_LEFT] = box[0]
    dict_box[Corner.TOP_RIGHT] = box[1]
    dict_box[Corner.BOTTOM_RIGHT] = box[2]
    dict_box[Corner.BOTTOM_LEFT] = box[3]
    return dict_keypoint, dict_box


def point_max_confidence(dict_keypoint):
    kp_coord_max, kp_type_max, kp_max_score = None, None, 0
    for k, v in dict_keypoint.items():
        if v['score'] > kp_max_score:
            kp_max_score = v['score']
            kp_coord_max = v['coord']
            kp_type_max = k
    return kp_coord_max, kp_type_max


def distance2point(p1: list, p2: list) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def rotate_corner(box_point, delta):
    result = {}
    if delta < 0:
        delta += 4
    for k, v in box_point.items():
        result[(k.value + delta) % 4] = v
    return result


def norm_box(keypoint, box, img):
    try:
        h, w = img.shape[0], img.shape[1]
    except:
        print("Don't find image")
    box_tly, box_tlx, box_bry, box_brx = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)
    orig_box = [[box_tlx, box_tly], [box_brx, box_tly], [box_brx, box_bry], [box_tlx, box_bry]]
    orig_keypoint = []
    if len(keypoint) != 0:
        y = [keypoint[i][0] * h for i in range(len(keypoint))]
        x = [keypoint[i][1] * w for i in range(len(keypoint))]
        for i in range(len(x)):
            orig_keypoint.append([int(x[i]), int(y[i])])
    return orig_box, orig_keypoint


def fill_the_keypoint(keypoint: list, box: list, keypoint_socre: list, img=None) -> dict:
    box, keypoint = norm_box(keypoint, box, img)
    dict_keypoint, dict_box = list_to_dict(keypoint, box, keypoint_socre)
    print(f"dict_keypoint: {dict_keypoint}")
    print(f"dict_box: {dict_box}")
    box_width = np.abs(dict_box[Corner.TOP_LEFT][0] - dict_box[Corner.TOP_RIGHT][0])
    box_height = np.abs(dict_box[Corner.BOTTOM_LEFT][1] - dict_box[Corner.TOP_LEFT][1])
    angle_delta = 0
    if box_height > box_width:
        angle_delta = -1
    kp_coord_max, kp_type_max = point_max_confidence(dict_keypoint)
    if kp_type_max is not None:
        c_coord_min = dict_box[Corner.TOP_LEFT]
        c_type_min = Corner.TOP_LEFT
        c_min_distance = distance2point(kp_coord_max, c_coord_min)
        for k, v in dict_box.items():
            d = distance2point(kp_coord_max, v)
            if d < c_min_distance:
                c_min_distance = d
                c_coord_min = v
                c_type_min = k
        angle_delta = kp_type_max.value - c_type_min.value

    box_candidate = rotate_corner(dict_box, angle_delta)
    for k in Corner:
        if k not in dict_keypoint:
            dict_keypoint[k] = {'coord': box_candidate[k]}
    return dict_keypoint

