import glob
import os
import xml.etree.ElementTree as ET
import cv2
from dateutil.parser import parser


def crop_box_infor(path_xml, path_img, path_output):
    tree = ET.parse(path_xml)
    root = tree.getroot()
    filename = root.find("filename").text
    file_img = f"{path_img}/{filename}"
    if os.path.exists(file_img):
        for obj in root.iter("object"):
            img = cv2.imread(file_img)
            label = obj.find("name").text
            x_min = int(obj.find("bndbox/xmin").text)
            y_min = int(obj.find("bndbox/ymin").text)
            x_max = int(obj.find("bndbox/xmax").text)
            y_max = int(obj.find("bndbox/ymax").text)
            img = img[y_min:y_max, x_min:x_max]
            if label == "EXPIRY" or label == "BIRTHDAY" or label == "ISSUE DATE":
                name = f"birthday_{filename}" if label == "BIRTHDAY" else f"expiry_{filename}"
                print(filename)
                cv2.imwrite(f"{path_output}/{name}", img)



if __name__ == "__main__":
    path_img = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped/cropped_back"
    path_xml = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped/cropped_back"
    path_output = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped/infor_cropped/DATE_0406"
    for file_xml in glob.glob(path_xml+"/*.xml"):
        crop_box_infor(file_xml, path_img, path_output)