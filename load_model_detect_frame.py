import glob
import time
import tensorflow as tf
import os
import numpy as np
import cv2
import warnings
from PIL import Image
from matplotlib import pyplot as plt
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils

warnings.filterwarnings('ignore')


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")

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
        keypoints = None
        keypoint_scores = None

        if "detection_keypoints" in detections:
            keypoints = detections['detection_keypoints']
            keypoint_scores = detections['detection_keypoint_scores']
        label_id_offset = 1
        image_np_with_detections = img_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            keypoints=keypoints,
            keypoint_edges=[[0, 1], [1, 2], [2, 3], [3, 0]],
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        print(f"Time {name} is: {time.time() - st}")
        cv2.imshow(name,image_np_with_detections)
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
    IMAGE_PATHS = [file for file in glob.glob(PATH + "/*")]
    PATH_TO_LABEL = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame/label_map.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL, use_display_name=True)
    print(f"Time load model is: {time.time() - start}")
    run()
