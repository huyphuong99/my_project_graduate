import glob
import xml.etree.ElementTree as ET
from file_detection.load_model_detect_frame import run
from object_detection.builders import model_builder
from object_detection.utils import label_map_util
from object_detection.utils import config_util
import tensorflow as tf
import os
import cv2


PATH_TO_LABEL_INFOR = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_infor/label_map.pbtxt"
PATH_TO_MODEL_INFOR = "/home/huyphuong/PycharmProjects/project_graduate/file_model_detect_infor"
PATH_TO_CONFIG_INFOR = PATH_TO_MODEL_INFOR + "/pipeline.config"
PATH_TO_CKPT_INFOR = PATH_TO_MODEL_INFOR + "/checkpoint"
NAME_MODEL_INFOR = "ckpt-31"
configs_inf = config_util.get_configs_from_pipeline_file(PATH_TO_CONFIG_INFOR)
model_config_inf = configs_inf['model']
detection_model_inf = model_builder.build(model_config=model_config_inf, is_training=False)
ckpt_inf = tf.compat.v2.train.Checkpoint(model=detection_model_inf)
ckpt_inf.restore(os.path.join(PATH_TO_CKPT_INFOR, NAME_MODEL_INFOR)).expect_partial()
category_index_infor = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_INFOR,
                                                                          use_display_name=True)

_dict = {1: "ID", 2: "ADDRESS", 3: "BIRTHDAY", 4: "NAME", 5: "TITLE", 6: "DOMICILE",
                  7: "COUNTRY", 8: "ETHNICITY", 9: "SEX", 10: "EXPIRY", 11: "ISSUE BY", 12: "ISSUE DATE",
                  13: "RELIGION"}

def create_xml(dict_image,_path,  _name, _folder, _db, path_output):
    img = cv2.imread(_path)
    h, w, d = img.shape
    data_annotation = ET.Element("annotation")
    folder = ET.SubElement(data_annotation, "folder")
    folder.text = f"{_folder}"
    file_name = ET.SubElement(data_annotation, "filename")
    file_name.text = f"{_name}"
    path = ET.SubElement(data_annotation,"path")
    path.text = f"{_path}"
    source = ET.SubElement(data_annotation, "source")
    data_base = ET.SubElement(source, "database")
    data_base.text = f"{_db}"
    size = ET.SubElement(data_annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = f"{w}"
    height = ET.SubElement(size, "height")
    height.text = f"{h}"
    depth = ET.SubElement(size,"depth")
    depth.text = f"{d}"
    segmented = ET.SubElement(data_annotation,"segmented")
    segmented.text  = f"{0}"
    for i in range(len(dict_image)):
        object = ET.SubElement(data_annotation,"object")
        name = ET.SubElement(object, "name")
        # print(dict_image[0]['class_idx'])
        name.text = f"{_dict[int(dict_image[i]['class_idx'])+1]}"
        pose = ET.SubElement(object, "pose")
        pose.text = "Unspecified"
        truncated = ET.SubElement(object, "truncated")
        truncated.text = f"{0}"
        difficult = ET.SubElement(object, "difficult")
        difficult.text = f"{0}"
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = f"{dict_image[i]['box'][0]}"
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = f"{dict_image[i]['box'][1]}"
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = f"{dict_image[i]['box'][2]}"
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text =  f"{dict_image[i]['box'][3]}"

    b_xml = ET.tostring(data_annotation)
    print(b_xml)
    with open(f"{path_output}/{_name.replace('jpg','xml')}", "wb") as f:
        f.write(b_xml)

if __name__ == "__main__":
    path_input = "/media/huyphuong/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped_new_cccd_270721/*"
    path = [file for file in glob.glob(sorted(path_input))]
    path_output = "/media/huyphuong/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped_new_cccd_270721"
    for i in range(len(path)):
        _name = os.path.basename(path[i])
        _folder = path[i].split("/")[-2]
        _db = "Unknown"
        obj = run(path[i], detection_model_inf, category_index_infor, _name)
        create_xml(obj, path[i], _name, _folder, _db, path_output)
