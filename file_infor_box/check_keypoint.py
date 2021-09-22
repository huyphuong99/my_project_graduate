from file_infor_box.crop_frame_from_img import Corner
import numpy as np
import cv2


def rotate_corner(box_points, delta):
    result = {}
    if delta < 0:
        delta = delta + 4
    for k, v in box_points.items():
        result[Corner((k.value + delta) % 4)] = v
    return result


def distance_2_points(p1, p2) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)



def _fill_keypoints(keypoints, box):
    box_candidate = {
        Corner.TOP_LEFT: (box[0], box[1]),
        Corner.TOP_RIGHT: (box[2], box[1]),
        Corner.BOTTOM_RIGHT: (box[2], box[3]),
        Corner.BOTTOM_LEFT: (box[0], box[3]),
    }
    angle_delta = 0
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_height > box_width:
        angle_delta = -1

    kp_coord_max = None
    kp_type_max = None
    kp_max_score = 0
    for k, v in keypoints.items():
        if v['score'] > kp_max_score:
            kp_max_score = v['score']
            kp_coord_max = v['coord']
            kp_type_max = k
    if kp_type_max is not None:

        c_coord_min = box_candidate[Corner.TOP_LEFT]
        c_type_min = Corner.TOP_LEFT
        c_min_distance = distance_2_points(kp_coord_max, c_coord_min)

        for k, v in box_candidate.items():
            d = distance_2_points(kp_coord_max, v)
            if d < c_min_distance:
                c_min_distance = d
                c_coord_min = v
                c_type_min = k

        angle_delta = kp_type_max.value - c_type_min.value

    box_candidate = rotate_corner(box_candidate, angle_delta)

    for k in Corner:
        if k not in keypoints:
            keypoints[k] = {'coord': box_candidate[k]}
    return keypoints

def _crop_img_with_point(img, keypoints, box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    tmp1 = max(width, height)
    tmp2 = min(width, height)
    width = tmp1
    height = tmp2

    keypoints = _fill_keypoints(keypoints, box)

    src = [
        keypoints[Corner.TOP_LEFT]['coord'],
        keypoints[Corner.TOP_RIGHT]['coord'],
        keypoints[Corner.BOTTOM_RIGHT]['coord'],
        keypoints[Corner.BOTTOM_LEFT]['coord'],
    ]

    src = np.array(src, dtype=np.float32)

    dst = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    m = cv2.getPerspectiveTransform(src, dst)
    wrarped = cv2.warpPerspective(img, m, (width, height))
    return wrarped

def _crop_with_keypoints(img, objects):
    for obj in objects:
        keypoints = obj['keypoints']
        box = obj['box']
        img_obj = _crop_img_with_point(img, keypoints, box)
        obj['obj_img'] = img_obj
    return objects

def _crop_without_keypoints(img, objects):
    for obj in objects:
        box = obj['box']
        img_obj = img[box[1]: box[3], box[0]: box[2]]
        obj['obj_img'] = img_obj
    return objects