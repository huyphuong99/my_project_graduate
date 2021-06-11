from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time
import cv2

from tensorflow.python.summary.summary import image

warnings.filterwarnings('ignore')


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


def load_img():
    for img_path in IMAGE_PATHS:
        img_np = load_image_into_numpy_array(img_path)
        input_tensor = tf.convert_to_tensor(np.expand_dims(img_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections] for key, value in detections.items()}
        detections["num_detections"] = num_detections
        detections["detection_classes"] = tf.cast(detections["detection_classes"], dtype=tf.int64)
        label_id_offset = 1
        image_np_with_detections = img_np.copy()
        print(detections['detection_boxes'])
        print(f"Time is: {time.time() - st}")
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False)
        plt.figure(figsize=(200, 240))
        plt.imshow(image_np_with_detections)
        plt.show()

if __name__ == "__main__":
    st = time.time()
    PATH_TO_MODEL_DIR = "./model/checkpoint"
    PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline_id.config"
    PATH_TO_CKPT = PATH_TO_MODEL_DIR
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt_model = "ckpt-4"
    ckpt.restore(os.path.join(PATH_TO_CKPT, ckpt_model)).expect_partial()

    IMAGE_PATHS = ["/home/huyphuong99/Desktop/CMND Nữ-05.jpg",
                   "/home/huyphuong99/Desktop/Trước 2.jpg"]
    PATH_TO_LABELS = "/home/huyphuong99/PycharmProjects/project_graduate/model/annotations/label_map_id.pbtxt"
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)
    load_img()
