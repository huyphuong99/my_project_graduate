import glob
import json
import os
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from file_detection.load_model_detect_frame import run
from file_infor_box.crop_frame_from_img import Corner

PATH_TO_LABEL_FRAME = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame/label_map.pbtxt"
PATH_TO_MODEL_DIR_FRAME = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_frame"
PATH_TO_CONFIG_FRAME = PATH_TO_MODEL_DIR_FRAME + "/pipeline.config"
PATH_TO_CKPT_FRAME = PATH_TO_MODEL_DIR_FRAME + "/checkpoint"
NAME_MODEL_FRAME = 'ckpt-21'

configs_fr = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG_FRAME)
model_config_fr = configs_fr['model']
detection_model_fr = model_builder.build(model_config=model_config_fr, is_training=False)
ckpt_fr = tf.compat.v2.train.Checkpoint(model=detection_model_fr)
ckpt_fr.restore(os.path.join(PATH_TO_CKPT_FRAME, NAME_MODEL_FRAME)).expect_partial()
category_index_frame = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_FRAME,
                                                                          use_display_name=True)


def create_json_file(object: dict, name, output_path):
    dict_label = {}
    name = name
    box = object["box"]
    keypoint = object["keypoints"]
    h, w, d = object["obj_img"].shape
    tl, tr, br, bl = list(keypoint[Corner.TOP_LEFT]["coord"]), list(keypoint[Corner.TOP_RIGHT]["coord"]), \
                     list(keypoint[Corner.BOTTOM_RIGHT]["coord"]), list(keypoint[Corner.BOTTOM_LEFT]["coord"])
    dict_label["version"] = "4.5.7"
    dict_label["flags"] = {}
    dict_label["shapes"] = []
    dict_label["imagePath"] = name
    dict_label["imageData"] = None
    dict_label["imageHeight"] = h
    dict_label["imageWidth"] = w
    dict_label["shapes"].append({})
    dict_label["shapes"][0]["label"] = "id"
    dict_label["shapes"][0]["points"] = [tl, tr, br, bl]
    dict_label["shapes"][0]["group_id"] = None
    dict_label["shapes"][0]["shape_type"] = "polygon"
    dict_label["shapes"][0]["flags"] = {}
    # print(dict_label)
    idx = name.index(".")
    with open(f"{os.path.join(output_path, name[:idx])}.json", 'w') as outfile:
        json.dump(dict_label, outfile)

if __name__ == "__main__":
    path_dir = "/media/huyphuong/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/new_cccd_270721"
    output_json = "/media/huyphuong/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/new_cccd_270721"
    paths = [file for file in sorted(glob.glob(os.path.join(path_dir, "*.jpg")))]
    # paths = [f"{path_dir}/06758_10364174_1812877.jpg"]
    for i in range(len(paths)):
        name = os.path.basename(paths[i])
        t = name.index(".")
        n = name[:t]
        print(name)
        if os.path.exists(f"{os.path.join(path_dir, n)}.json"):
            continue
        object = run(paths[i], detection_model_fr, category_index_frame)
        if len(object) == 0:
            continue
        create_json_file(object[0], name, output_json)


