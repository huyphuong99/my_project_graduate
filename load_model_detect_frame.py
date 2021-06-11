import glob
import math
import time
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


@tf.function
def detect_fn(image):
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


def run():
    for img_path in IMAGE_PATHS:
        st = time.time()
        name = os.path.basename(img_path)
        img_np = load_image_into_numpy_array(img_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)
        scores = detections['detection_scores']
        box1 = detections['detection_boxes']
        keypoints_1 = None
        keypoint_scores = None
        if "detection_keypoints" in detections:
            keypoints_1 = detections['detection_keypoints']
            keypoint_scores = detections['detection_keypoint_scores']

        keypoints_i = []
        thres_box = .3
        box = []
        for i in range(len(scores)):
            if scores[i] > thres_box:
                box.append(box1[i].tolist())
                keypoints_i.append(keypoints_1[i])
        keypoints_all = []
        keypoint_score_all = []
        box_all = []
        for k in range(len(keypoints_i)):
            keypoints = []
            #NOTE: box and keypoint has coordinate is [ymin, xmin, ymax, xmax], [top_left_y, top_left_x,..., bottom_left_y, top_left_x]
            y = [keypoints_i[k][i][0] * (img_np.shape)[0] for i in range(len(keypoints_i[0]))]
            x = [keypoints_i[k][i][1] * (img_np.shape)[1] for i in range(len(keypoints_i[0]))]
            box_tly, box_tlx, box_bry, box_brx = int(box[k][0] * (img_np.shape)[0]), int(
                box[k][1] * (img_np.shape)[1]), int(box[k][
                                                        2] * (img_np.shape)[0]), int(box[k][3] * (img_np.shape)[1])
            orig_box = [[box_tlx, box_tly], [box_brx, box_tly], [box_brx, box_bry], [box_tlx, box_bry]]
            box_all.append(orig_box)
            for i in range(len(x)):
                keypoints.append([int(x[i]), int(y[i])])
            #Check keypoint enough or not, if it have not kepoints, take a coordinate of box -> coordinate
            keypoints = check_has_keypoint(orig_box, keypoints, keypoint_scores[k].tolist())
            keypoints_all.append(keypoints)
            keypoint_score_all.append((keypoint_scores[k].tolist()))

        print(f"keypoint score: {keypoint_score_all}")
        print(f"keypoint: {keypoints_all}")
        print(f"box: {box_all}")
        label_id_offset = 1
        image_np_with_detections = img_np.copy()
        image = \
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                keypoints=keypoints_1,
                keypoint_scores=keypoint_scores,
                keypoint_edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                line_thickness=1,
                agnostic_mode=False)
        cv2.imshow(name, image)
        cv2.waitKey()
        print(f"Time {name} is: {time.time() - st}")
        for keypoints in keypoints_all:
            wraped = four_point_transform(image_np_with_detections, np.array(keypoints, dtype="float32"))
            resize = 100
            wraped = cv2.resize(wraped, (int(wraped.shape[1] * resize / 100), int(wraped.shape[0] * resize/ 100)))
            cv2.imshow(name, wraped)
            cv2.waitKey()


if __name__ == "__main__":
    start = time.time()
    PATH_TO_MODEL_DIR = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame"
    PATH_TO_CONFIG = PATH_TO_MODEL_DIR + "/pipeline.config"
    PATH_TO_CKPT = PATH_TO_MODEL_DIR + "/checkpoint"
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-28')).expect_partial()
    PATH = "/home/huyphuong/PycharmProjects/project_graduate/input/input_origin_image"
    IMAGE_PATHS = ["/home/huyphuong/Desktop/material/test/chungminh11.jpg"]
    PATH_TO_LABEL = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame/label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL, use_display_name=True)
    print(f"Time load model is: {time.time() - start}")
    run()
