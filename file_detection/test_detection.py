import glob
import time
import tensorflow as tf
import os
from file_infor_box.crop_frame_from_img import Corner
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import config_util
import numpy as np
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils
from file_infor_box.check_keypoint import _crop_with_keypoints
import matplotlib
import matplotlib.pyplot as plt
import warnings
matplotlib.use("tkagg")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


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


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def detect_fn(image, detection_model):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

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
            min_score_thresh=.80,
            line_thickness=3,
            agnostic_mode=False,
            skip_labels=False,
            skip_scores=False)
    # resize = 50
    # image = cv2.resize(image, (int(image.shape[1] * resize / 100), int(image.shape[0] * resize / 100)))
    plt.figure(figsize=(8,6))
    plt.imshow(image)
    plt.title("image")
    plt.show()
    # cv2.imshow("name",image)
    # cv2.waitKey()

    # cv2.imshow(name, image)
    # cv2.waitKey()


def run(IMAGE_PATHS, detection_model, category_index, name=None):
    if os.path.exists(f"{IMAGE_PATHS}"):
        img_np = load_image_into_numpy_array(IMAGE_PATHS)
        name = os.path.basename(IMAGE_PATHS)
    else:
        img_np = IMAGE_PATHS
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
        draw_box_to_image(img_np, detections, category_index, keypoints_1, keypoint_scores, name)
        keypoints_1[:, :, 0] *= h_img
        keypoints_1[:, :, 1] *= w_img
    else:
        draw_box_to_image(img_np, detections, category_index, keypoints_1, keypoint_scores, name)
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
    # else:
    #     objects = _crop_without_keypoints(img_np, objects)
    return objects


def main_detection(path, detection_model_fr, category_index_frame, name):
    IMAGE_PATHS_FRAME = path
    object_fr = run(IMAGE_PATHS_FRAME, detection_model_fr, category_index_frame,name)



if __name__=="__main__":
    # CONFIG MODEL DETECTION
    PATH_TO_LABEL_FRAME = "/home/huyphuong99/Desktop/material/model_test/model_detection_id_passport_vr/label_map.pbtxt"
    PATH_TO_MODEL_DIR_FRAME = "/home/huyphuong99/Desktop/material/model_test/model_detection_id_passport_vr"
    PATH_TO_CONFIG_FRAME = PATH_TO_MODEL_DIR_FRAME + "/pipeline.config"
    PATH_TO_CKPT_FRAME = PATH_TO_MODEL_DIR_FRAME + "/checkpoint"
    NAME_MODEL_FRAME = 'ckpt-29'

    # CONFIG DETECTION FRAME
    configs_fr = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG_FRAME)
    model_config_fr = configs_fr['model']
    detection_model_fr = model_builder.build(model_config=model_config_fr, is_training=False)
    ckpt_fr = tf.compat.v2.train.Checkpoint(model=detection_model_fr)
    ckpt_fr.restore(os.path.join(PATH_TO_CKPT_FRAME, NAME_MODEL_FRAME)).expect_partial()
    category_index_frame = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_FRAME,
                                                                              use_display_name=True)

    # paths = ["/home/huyphuong99/Desktop/material/test/passport/004566.png"]
    paths = glob.glob("/home/huyphuong99/Desktop/material/test/passport/*")

    for path in paths:
        name = os.path.basename(path)
        main_detection(path, detection_model_fr, category_index_frame, name)

