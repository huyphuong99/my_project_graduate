import math
import tensorflow as tf
import os
import numpy as np
import cv2
import warnings
from PIL import Image
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from file_training_object_detection.crop_box_from_image import four_point_transform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


# @tf.function
def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def initi_box(box, coor_list):
    dict_box = {}
    for i in range(len(coor_list)):
        dict_box[coor_list[i]] = box[i]
    return dict_box


def distance_to_box(keypoint, box):
    dist_min = 1000
    idx = None
    for i in range(len(box)):
        dist = math.sqrt((keypoint[0] - box[i][0]) ** 2 + (keypoint[1] - box[i][1]) ** 2)
        if dist < dist_min:
            dist_min = dist
            idx = i
    return box[idx]


def check_has_keypoint(box, keypoints, keypoint_score, thres_keypoint=.3):
    coor_list = ["top_left", "top_right", "bot_right", "bot_left"]
    dict_box = initi_box(box, coor_list)
    dict_keypoint = initi_box(keypoints, coor_list)

    for i in range(len(keypoint_score)):
        if keypoint_score[i] < thres_keypoint or keypoint_score[i] is None:
            dict_keypoint[coor_list[i]] = distance_to_box(dict_keypoint[coor_list[i]], box)
    keypoint = []
    for key, value in dict_keypoint.items():
        keypoint.append(value)
    return keypoint


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
            min_score_thresh=.30,
            line_thickness=1,
            agnostic_mode=False)
    resize = 50
    image = cv2.resize(image, (int(image.shape[1] * resize / 100), int(image.shape[0] * resize / 100)))
    cv2.imshow(name, image)
    cv2.waitKey()


def check_keypoint(scores, box1, keypoints_1, img_np, keypoint_scores, label):
    keypoints_i = []
    thres_box = .3
    box = []
    label_ = []
    for i in range(len(scores)):
        if scores[i] > thres_box:
            box.append(box1[i].tolist())
            label_.append(label[i].tolist())
            if keypoints_1 is not None:
                keypoints_i.append(keypoints_1[i])
    keypoints_all = []
    keypoint_score_all = []
    box_all = []
    for k in range(len(box)):
        box_tly, box_tlx, box_bry, box_brx = int(box[k][0] * (img_np.shape)[0]), int(
            box[k][1] * (img_np.shape)[1]), int(box[k][2] * (img_np.shape)[0]), int(box[k][3] * (img_np.shape)[1])
        orig_box = [[box_tlx, box_tly], [box_brx, box_tly], [box_brx, box_bry], [box_tlx, box_bry]]
        box_all.append(orig_box)
        keypoints = []
        # NOTE: box and keypoint has coordinate is [ymin, xmin, ymax, xmax], [top_left_y, top_left_x,..., bottom_left_y, top_left_x]
        if len(keypoints_i) != 0:
            y = [keypoints_i[k][i][0] * (img_np.shape)[0] for i in range(len(keypoints_i[0]))]
            x = [keypoints_i[k][i][1] * (img_np.shape)[1] for i in range(len(keypoints_i[0]))]
            for i in range(len(x)):
                keypoints.append([int(x[i]), int(y[i])])
            # Check keypoint enough or not, if it have not kepoints, take a coordinate of box -> coordinate
            keypoints = check_has_keypoint(orig_box, keypoints, keypoint_scores[k].tolist())
            keypoints_all.append(keypoints)
            keypoint_score_all.append((keypoint_scores[k].tolist()))
    if keypoints_1 is None:
        return box_all, label_
    else:
        return keypoints_all, label_


def run(IMAGE_PATHS, PATH_TO_CONFIG, PATH_TO_CKPT, NAME_MODEL, PATH_TO_LABEL):
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, NAME_MODEL)).expect_partial()
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL, use_display_name=True)
    if os.path.exists(f"{IMAGE_PATHS}"):
        img_np = load_image_into_numpy_array(IMAGE_PATHS)
        global name
        name = os.path.basename(IMAGE_PATHS)
    else:
        img_np = IMAGE_PATHS
    input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections["num_detections"] = num_detections
    detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
    scores = detections['detection_scores']
    box1 = detections['detection_boxes']
    label = detections["detection_classes"]
    keypoints_1 = None
    keypoint_scores = None
    if "detection_keypoints" in detections:
        keypoints_1 = detections['detection_keypoints']
        keypoint_scores = detections['detection_keypoint_scores']
    image_np_with_detections = img_np.copy()
    draw_box_to_image(img_np, detections, category_index, keypoints_1, keypoint_scores, name)
    keypoints_all, label = check_keypoint(scores, box1, keypoints_1, img_np, keypoint_scores, label)
    image_frame = []
    for keypoints in keypoints_all:
        wraped = four_point_transform(image_np_with_detections, np.array(keypoints, dtype="float32"))
        # resize = 100
        # wraped = cv2.resize(wraped, (int(wraped.shape[1] * resize / 100), int(wraped.shape[0] * resize / 100)))
        image_frame.append(wraped)
    return image_frame, label, keypoints_all


def assign2line(value):
    # print(value)
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
        value2row, value_row_down = assign2line(value)
        value = value2row if (len(value_row_down) > 1) else sorted(value, key=(lambda box: box[1][0]))
        for i in range(len(value)):
            if f"{dict_label[key + 1]}" not in dict_image:
                dict_image[f"{dict_label[key + 1]}"] = [value[i][0]]
            else:
                dict_image[f"{dict_label[key + 1]}"].append(value[i][0])
            # cv2.imshow(f"{dict_label[key + 1]}", value[i][0])
            # cv2.waitKey()
    # print(dict_image)
    return dict_image


def main_detection(path):
    PATH_TO_LABEL_FRAME = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame/label_map.pbtxt"
    PATH_TO_MODEL_DIR_FRAME = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame"
    IMAGE_PATHS_FRAME = path
    PATH_TO_CONFIG_FRAME = PATH_TO_MODEL_DIR_FRAME + "/pipeline.config"
    PATH_TO_CKPT_FRAME = PATH_TO_MODEL_DIR_FRAME + "/checkpoint"
    NAME_MODEL_FRAME = 'ckpt-28'
    PATH_TO_LABEL_INFOR = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_infor/label_map.pbtxt"
    PATH_TO_MODEL_INFOR = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_infor"
    PATH_TO_CONFIG_INFOR = PATH_TO_MODEL_INFOR + "/pipeline.config"
    PATH_TO_CKPT_INFOR = PATH_TO_MODEL_INFOR + "/checkpoint"
    NAME_MODEL_INFOR = "ckpt-31"
    IMAGE_CROPPED_FRAME, label_frame, boxes_frame = run(IMAGE_PATHS_FRAME, PATH_TO_CONFIG_FRAME, PATH_TO_CKPT_FRAME,
                                                        NAME_MODEL_FRAME, PATH_TO_LABEL_FRAME)
    image_infor, label_infor, boxes_infor = run(IMAGE_CROPPED_FRAME[0], PATH_TO_CONFIG_INFOR, PATH_TO_CKPT_INFOR,
                                                NAME_MODEL_INFOR, PATH_TO_LABEL_INFOR)
    dict_image = crop_image_infor(image_infor, boxes_infor, label_infor)
    return dict_image


if __name__ == "__main__":
    x = main_detection()
