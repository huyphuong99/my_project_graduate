from builtins import dict, print, object
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
import cv2
import warnings
from PIL import Image
from file_infor_box.crop_frame_from_img import Corner
from object_detection.utils import visualization_utils as viz_utils
from file_infor_box.check_keypoint import _crop_without_keypoints, _crop_with_keypoints

matplotlib.use("tkagg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

output_front_cropped = "/home/huyphuong99/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_image/cropped_id_front/cropped_image_front"
output_back_cropped = "/home/huyphuong99/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_image/cropped_id_back/cropped_image_back"
output_front_infor = "/home/huyphuong99/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_image/cropped_id_front/cropped_infor_front"
output_back_infor = "/home/huyphuong99/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_image/cropped_id_back/cropped_infor_back"


class define:
    confidence_score = 0.4


def nms(objects, overlap_thresh):
    if len(objects) == 0:
        return []
    boxes = np.array([x['box'] for x in objects])
    scores = np.array([x['score'] for x in objects])

    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Picked bounding boxes
    picked_objects = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    order = np.argsort(scores)
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_objects.append(objects[index])

        # Compute cordinates of intersection-over-union (IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < overlap_thresh)
        order = order[left]
    return picked_objects


# @tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def draw_box_to_image(image_np_with_detections, detections, category_index, keypoints, keypoint_scores, name,
                      label_id_offset=1):
    image = \
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.40,
            line_thickness=3,
            agnostic_mode=False,
            skip_labels=False,
            skip_scores=False)
    # resize = 50
    # image = cv2.resize(image, (int(image.shape[1] * resize / 100), int(image.shape[0] * resize / 100)))
    # plt.figure(figsize=(8,6))
    # plt.imshow(image)
    # plt.title("image")
    # plt.show()
    # cv2.imshow("name",image)
    # cv2.waitKey()

    cv2.imshow(name, image)
    cv2.waitKey()


def run(IMAGE_PATHS, detection_model, category_index, name=None):
    image_cropped = None
    if os.path.exists(f"{IMAGE_PATHS}"):
        img_np = load_image_into_numpy_array(IMAGE_PATHS)
        name = os.path.basename(IMAGE_PATHS)
    else:
        img_np = IMAGE_PATHS
        image_cropped = IMAGE_PATHS
    # print(img_np)
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    # print(detections["detection_classes"])
    scores = detections['detection_scores']
    box1 = detections['detection_boxes']
    label = detections["detection_classes"]
    h_img, w_img = img_np.shape[0], img_np.shape[1]
    has_keypoint = "detection_keypoints" in detections
    label_ = label.astype(np.int32).tolist()
    keypoints_1, keypoint_scores = None, None
    if has_keypoint:
        keypoint_scores = detections['detection_keypoint_scores']
        keypoints_1 = detections['detection_keypoints']
        # draw_box_to_image(img_np, detections, category_index, keypoints_1, keypoint_scores, name)
        keypoints_1[:, :, 0] *= h_img
        keypoints_1[:, :, 1] *= w_img
    else:
        pass
        # draw_box_to_image(img_np, detections, category_index, keypoints_1, keypoint_scores, name)
    box1[:, 0] *= h_img
    box1[:, 1] *= w_img
    box1[:, 2] *= h_img
    box1[:, 3] *= w_img
    box1 = box1.astype(np.int32)
    objects = []
    for i in range(box1.shape[0]):
        if scores[i] is None or scores[i] > define.confidence_score:
            box = tuple(box1[i].tolist())
            box = (box[1], box[0], box[3], box[2])
            score = scores[i]
            label_idx = label_[i]
            label = str(label_idx)
            keypoint_objs = {}
            if has_keypoint:
                if keypoint_scores[i, 0] > define.confidence_score:
                    keypoint_objs[Corner.TOP_LEFT] = {'coord': (int(keypoints_1[i, 0, 1]), int(keypoints_1[i, 0, 0])),
                                                      'score': keypoint_scores[i, 0]}
                if keypoint_scores[i, 1] > define.confidence_score:
                    keypoint_objs[Corner.TOP_RIGHT] = {'coord': (int(keypoints_1[i, 1, 1]), int(keypoints_1[i, 1, 0])),
                                                       'score': keypoint_scores[i, 1]}
                if keypoint_scores[i, 2] > define.confidence_score:
                    keypoint_objs[Corner.BOTTOM_RIGHT] = {
                        'coord': (int(keypoints_1[i, 2, 1]), int(keypoints_1[i, 2, 0])),
                        'score': keypoint_scores[i, 2]}
                if keypoint_scores[i, 3] > define.confidence_score:
                    keypoint_objs[Corner.BOTTOM_LEFT] = {
                        'coord': (int(keypoints_1[i, 3, 1]), int(keypoints_1[i, 3, 0])),
                        'score': keypoint_scores[i, 3]}
            objects.append({
                'score': score,
                'box': box,
                'class_idx': label_idx,
                'class_name': label,
                'keypoints': keypoint_objs

            })
    objects = nms(objects, define.confidence_score)
    if has_keypoint:
        objects = _crop_with_keypoints(img_np, objects)
    else:
        objects = _crop_without_keypoints(img_np, objects)
    return objects, image_cropped


def assign2line(value):
    value = sorted(value, key=(lambda box: box[1][1]))
    value_row_up = []
    value_row_down = []
    for i in range(len(value)):
        thres_down = int((value[0][1][1] + value[-1][1][1]) / 2) + 5
        if value[i][1][1] < thres_down:
            row_up = value[i]
            value_row_up.append(row_up)
        else:
            row_down = value[i]
            value_row_down.append(row_down)
    value_row_up = sorted(value_row_up, key=(lambda box: box[1][0]))
    value_row_down = sorted(value_row_down, key=(lambda box: box[1][0]))
    return value_row_up + value_row_down, value_row_down


def crop_image_infor(image_infor, boxes_infor, label_infor):
    boxes_infor = [boxes_infor[i][0] for i in range(len(boxes_infor))]
    dict_label = {1: "ID", 2: "ADDRESS", 3: "BIRTHDAY", 4: "NAME", 5: "TITLE", 6: "DOMICILE",
                  7: "COUNTRY", 8: "ETHNICITY", 9: "SEX", 10: "EXPIRY", 11: "ISSUE BY", 12: "ISSUE DATE",
                  13: "RELIGION"}
    dict_ = {}
    for i in range(len(label_infor)):
        if label_infor[i] in dict_:
            dict_[label_infor[i]].append([image_infor[i], boxes_infor[i]])
        else:
            dict_[label_infor[i]] = [[image_infor[i], boxes_infor[i]]]
    dict_image = {}
    for key, value in dict_.items():
        key = int(key)
        value2row, value_row_down = assign2line(value)
        value = value2row if (len(value_row_down) > 1) else sorted(value, key=(lambda box: box[1][0]))
        for i in range(len(value)):
            if f"{dict_label[key + 1]}" not in dict_image:
                dict_image[f"{dict_label[key + 1]}"] = [value[i][0]]
            else:
                dict_image[f"{dict_label[key + 1]}"].append(value[i][0])
    # print(dict_image)
    return dict_image


# coordinate = [(x_min, y_min), (x_max, y_max)]
def crop_img(image, coordinate, ratio=1 / 30.0):
    x_max, y_max = coordinate[1]
    x_min, y_min = coordinate[0]
    w_box = x_max - x_min
    h_box = y_max - y_min
    r = w_box / h_box
    x_min_new = x_min - int(ratio * w_box)
    y_min_new = y_min - int(r * ratio * h_box)
    x_max_new = x_max + int(ratio * w_box)
    y_max_new = y_max + int(r * ratio * h_box)
    img = image[y_min_new: y_max_new, x_min_new: x_max_new]
    return img


# coordinates
# [[(x_min, y_min), (x_max, y_max)],[]...]
def detect_box(raw_img: np.ndarray, boxes: list[tuple], ratio):
    # raw_img = cv2.imread(path)
    # boxes = sorted(coordinates, key=lambda x: x[1])
    label = {0: "id", 1: "fullname", 2: "birthday"}
    dict_label = {}
    for i, box in enumerate(boxes):
        dict_label[i] = crop_img(raw_img, box, ratio)
        cv2.imshow(f"{label[i]}", dict_label[i])
        cv2.waitKey()
    return dict_label



def crop_image_check_fraud(image_cr, box_info, label, ratio):
    dict_label = {1: "ID", 2: "ADDRESS", 3: "BIRTHDAY", 4: "NAME", 5: "TITLE", 6: "DOMICILE",
                  7: "COUNTRY", 8: "ETHNICITY", 9: "SEX", 10: "EXPIRY", 11: "ISSUE BY", 12: "ISSUE DATE",
                  13: "RELIGION"}
    info = {}
    for i in range(len(label)):
        lb = dict_label[int(label[i])+1]
        if lb == "ID":
            info[lb] = box_info[i]
        elif lb == "BIRTHDAY":
            info[lb] = box_info[i]
        elif lb == "NAME":
            if lb not in info:
                info[lb] = [box_info[i]]
            else:
                info[lb].append(box_info[i])
    t = sorted(info["NAME"], key=lambda x: x[0][0])
    info["NAME"] = [t[0][0], t[-1][-1]]
    info["image"] = image_cr
    print(info)
    list_info = [info["ID"],info["NAME"], info["BIRTHDAY"]]
    temp = detect_box(image_cr, list_info, ratio)
    return temp


def explore_dict_image(dict_image: dict, output_path):
    # print(dict_image)
    name = ""
    for key, value in dict_image.items():
        for i in range(len(value)):
            cv2.imwrite(f"{output_path}/{key}/{i}_{name}", value[i])


def main_detection(path, detection_model_fr, category_index_frame, detection_model_inf, category_index_infor, name, ratio):
    IMAGE_PATHS_FRAME = path
    object_fr, _ = run(IMAGE_PATHS_FRAME, detection_model_fr, category_index_frame, name)
    image_inf = object_fr[0]['obj_img']
    object_inf, img_cr = run(image_inf, detection_model_inf, category_index_infor, name)
    image_infor, boxes_infor, label_infor = [], [], []
    check_box = []
    labels = []
    for obj in object_inf:
        img = obj['obj_img']
        box = obj['box']
        box_temp = [(box[0], box[1]), (box[2], box[3])]  # [(x_min, y_min), (x_max, y_max)]
        box = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
        lb = obj['class_name']
        # print(lb, type(lb))
        # _dict = {1: "ID", 2: "ADDRESS", 3: "BIRTHDAY", 4: "NAME", 5: "TITLE", 6: "DOMICILE",
        #          7: "COUNTRY", 8: "ETHNICITY", 9: "SEX", 10: "EXPIRY", 11: "ISSUE BY", 12: "ISSUE DATE",
        #          13: "RELIGION"}
        image_infor.append(img)
        boxes_infor.append(box)
        label_infor.append(lb)
        check_box.append(box_temp)
        labels.append(lb)
    # print(check_box, "\n", labels)
    dict_image = crop_image_infor(image_infor, boxes_infor, label_infor)
    image_check_fraud = crop_image_check_fraud(img_cr, check_box, labels, ratio)
    print(image_check_fraud)
    return dict_image


if __name__ == "__main__":
    x = main_detection()
