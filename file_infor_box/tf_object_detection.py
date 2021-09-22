from typing import Dict, Tuple, Any, List

import cv2
import numpy as np



class Corner(Enum):
    order = 'TOP_LEFT TOP_RIGHT BOTTOM_RIGHT BOTTOM_LEFT'
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3


def rotate_corner(box_points: Dict[Corner, Tuple], delta: int) -> Dict[Corner, Tuple]:
    result = {}
    if delta < 0:
        delta = delta + 4
    for k, v in box_points.items():
        result[Corner((k.value + delta) % 4)] = v
    return result


def distance_2_points(p1: Tuple, p2: Tuple) -> float:
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)




class TFObjectDetection(BaseModelTF):
    def __init__(self,
                 service_host: str,
                 service_port: int,
                 model_name: str,
                 version: int = None,
                 model_input_size: tuple = (512, 512),
                 class_names: list = None,
                 confidence_threshold: float = 0.2,
                 keypoint_confidence_threshold: float = 0.2,
                 iou_threshold: float = 0.2,
                 options: list = [],
                 timeout: int = 5,
                 debug: bool = False):

        super(TFObjectDetection, self).__init__(
            service_host=service_host,
            service_port=service_port,
            model_name=model_name,
            version=version,
            options=options,
            timeout=timeout,
            debug=debug
        )
        self._model_input_size = model_input_size
        self._class_names = class_names
        self._confidence_threshold = confidence_threshold
        self._keypoint_confidence_threshold = keypoint_confidence_threshold
        self._iou_threshold = iou_threshold

    def _pre_process(self, x: np.ndarray) -> Tuple[Dict[Any, np.ndarray], dict]:
        if x is None or not isinstance(x, np.ndarray):
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input must be not None."))

        try:
            h, w, _ = x.shape
            ih, iw = self._model_input_size

            scale = min(iw / w, ih / h)
            nw, nh = int(scale * w), int(scale * h)
            image_resized = cv2.resize(x, (nw, nh))
            image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

            if len(image_resized.shape) == 3:
                image_resized = np.expand_dims(image_resized, axis=0)
            image_resized = image_resized.astype(np.uint8)

        except Exception as e:
            self.LOGGER.exception(e)
            raise InputFormatException("{}: {}".format(self.__class__.__name__, "input is wrong format"))
        return {
                   self._input_signature_key[0][0]: image_resized
               }, {'image_size': (h, w), 'image': x}

    def _post_process(self, predict_response: dict, params_process: dict) -> Dict:
        h_img, w_img = params_process['image_size']
        img = params_process['image']

        num_dectections = predict_response['num_detections']  # [N]
        boxes = predict_response['detection_boxes']  # [ymin, xmin, ymax, xmax].
        classes = predict_response['detection_classes']  # [N]
        scores = predict_response['detection_scores']  # [N]
        # multiclass_scores = predict_response['detection_multiclass_scores']
        if 'detection_boxes_strided' in predict_response:
            boxes_strided = predict_response['detection_boxes_strided']

        has_keypoints = 'detection_keypoints' in predict_response
        if has_keypoints:
            keypoints = predict_response['detection_keypoints']  # [1, N, 17, 2]
            keypoint_scores = predict_response['detection_keypoint_scores']  # [1, N, 17]
            keypoints[:, :, :, :1] *= h_img
            keypoints[:, :, :, 1:] *= w_img
            keypoints = keypoints.astype(np.int32)

        boxes[:, :, 0] *= h_img
        boxes[:, :, 1] *= w_img
        boxes[:, :, 2] *= h_img
        boxes[:, :, 3] *= w_img
        boxes = boxes.astype(np.int32)

        classes = classes.astype(np.int32) - 1

        objects = []

        for i in range(boxes.shape[0]):
            for j in range(boxes.shape[1]):
                if scores[i] is None or scores[i] > self._confidence_threshold:
                    box = tuple(boxes[i][j].tolist())
                    box = (box[1], box[0], box[3], box[2])
                    score = scores[i]
                    class_idx = classes[i][j]
                    class_name = str(class_idx)
                    if self._class_names is not None:
                        class_name = self._class_names[class_idx]

                    keypoint_objs = {}
                    if has_keypoints:
                        if keypoint_scores[i, j, 0] > self._keypoint_confidence_threshold:
                            keypoint_objs[Corner.TOP_LEFT] = {
                                'coord': (keypoints[i, j, 0, 1], keypoints[i, j, 0, 0]),
                                'score': keypoint_scores[i, j, 0]
                            }
                        if keypoint_scores[i, j, 1] > self._keypoint_confidence_threshold:
                            keypoint_objs[Corner.TOP_RIGHT] = {
                                'coord': (keypoints[i, j, 1, 1], keypoints[i, j, 1, 0]),
                                'score': keypoint_scores[i, j, 1]
                            }
                        if keypoint_scores[i, j, 2] > self._keypoint_confidence_threshold:
                            keypoint_objs[Corner.BOTTOM_RIGHT] = {
                                'coord': (keypoints[i, j, 2, 1], keypoints[i, j, 2, 0]),
                                'score': keypoint_scores[i, j, 2]
                            }
                        if keypoint_scores[i, j, 3] > self._keypoint_confidence_threshold:
                            keypoint_objs[Corner.BOTTOM_LEFT] = {
                                'coord': (keypoints[i, j, 3, 1], keypoints[i, j, 3, 0]),
                                'score': keypoint_scores[i, j, 3]
                            }

                    objects.append({
                        'score': score,
                        'box': box,
                        'class_idx': class_idx,
                        'class_name': class_name,
                        'keypoints': keypoint_objs

                    })
        #             print("{}: {}".format(class_name, score))
        #             if self._debug:
        #                 img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
        #                 img = cv2.putText(img, str(class_name), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
        #                 for k, v in keypoint_objs.items():
        #                     img = cv2.circle(img, v['coord'], 3, (0, 0, 255))
        #                     img = cv2.putText(img, str(k), v['coord'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        # cv2.imshow(f'{j}', img)
        # cv2.waitKey(0)
        objects = nms(objects, self._iou_threshold)
        if has_keypoints:
            objects = self._crop_with_keypoints(img, objects)
        else:
            objects = self._crop_without_keypoints(img, objects)

        scores = []
        boxes = []
        class_indices = []
        class_names = []
        keypoints = []
        obj_imgs = []
        for obj in objects:
            scores.append(obj['score'])
            boxes.append(obj['box'])
            class_indices.append(obj['class_idx'])
            class_names.append(obj['class_name'])
            keypoints.append(obj['keypoints'])
            obj_imgs.append(obj['obj_img'])

        return {
            'scores': scores,
            'boxes': boxes,
            'class_indices': class_indices,
            'class_names': class_names,
            'keypoints': keypoints,
            'obj_imgs': obj_imgs
        }

    def _normalize_output(self, post_processed: dict, params: dict) -> Dict[str, List]:
        results = {
            "scores": [],
            "boxes": [],
            "class_names": [],
            # "class_indices": []
        }

        type_output = 'meta'
        if 'type' in params:
            type_output = params['type']
        if type_output == 'image':
            results['obj_imgs'] = []

        if 'boxes' not in post_processed or 'class_names' not in post_processed:
            return results
        scores = post_processed['scores']
        boxes = post_processed['boxes']
        class_indices = post_processed['class_indices']
        class_names = post_processed['class_names']
        keypoints = post_processed['keypoints']
        obj_imgs = post_processed['obj_imgs']

        results['scores'] = scores
        results['boxes'] = boxes
        results['class_names'] = class_names
        # results['class_indices'] = class_indices

        if type_output == 'image':
            results['obj_imgs'] = obj_imgs
        return results

    def _crop_without_keypoints(self, img: np.ndarray, objects: List[Dict]) -> List[Dict]:
        for obj in objects:
            box = obj['box']
            img_obj = img[box[1]: box[3], box[0]: box[2]]
            obj['obj_img'] = img_obj
        return objects

    def _crop_with_keypoints(self, img: np.ndarray, objects: List[Dict]) -> List[Dict]:
        for obj in objects:
            keypoints = obj['keypoints']
            box = obj['box']
            class_name = obj['class_name']

            img_obj = self._crop_img_with_point(img, keypoints, box)
            obj['obj_img'] = img_obj
        return objects

    def _crop_img_with_point(self, img: np.ndarray, keypoints: Dict, box: List[int]) -> np.ndarray:
        width = box[2] - box[0]
        height = box[3] - box[1]
        tmp1 = max(width, height)
        tmp2 = min(width, height)
        width = tmp1
        height = tmp2

        keypoints = self._fill_keypoints(keypoints, box)

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

        # m = cv2.getPerspectiveTransform(src, dst)

    def _fill_keypoints(self, keypoints: dict, box: List[int]) -> Dict:
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


if __name__ == '__main__':
    import glob


    def test(img):
        res = model.predict(img, params={'type': 'image'})
        for k, v in res.items():
            for i, obj in enumerate(v):
                if isinstance(v[0], np.ndarray):
                    cv2.imshow("{}_{}".format(k, i), obj)
                else:
                    # print("{}_{}: {}".format(k, i, obj))
                    pass
        cv2.waitKey(0)


    class_names = ["other"] * 80
    class_names[4] = "motor"

    model = TFObjectDetection("172.16.1.48", 8500, 'coco_detection', debug=True, confidence_threshold=0.3,
                              class_names=class_names, options=[('grpc.max_receive_message_length', 100 * 1024 * 1024)])
    for file in glob.glob("/home/thiennt/Desktop/id_images/vehicle/*.jpg"):
        img = cv2.imread(file)
        test(img)
