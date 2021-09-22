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
        i = 0
        for obj in root.iter("object"):
            try:
                i += 1
                img = cv2.imread(file_img)
                label = obj.find("name").text
                x_min = int(obj.find("bndbox/xmin").text)
                y_min = int(obj.find("bndbox/ymin").text)
                x_max = int(obj.find("bndbox/xmax").text)
                y_max = int(obj.find("bndbox/ymax").text)
                img = img[y_min:y_max, x_min:x_max]
                path = f"{path_output}/{label}/{label}_{i}_{filename}"
                cv2.imwrite(path, img)
            except:
                print(filename)


if __name__ == "__main__":
    path_img = "/media/huyphuong/huyphuong99/tima/project/id/project_tima/info_id_do_an/data_raw/raw_image/raw_new_image/cropped_new_cccd_270721"
    path_xml = ""
    path_output = "/media/huyphuong/huyphuong99/tima/project/passport/raw_data/cropped/path_cropped_inf"
    i = 0
    for file_xml in glob.glob(path_xml + "/*.xml"):
        i += 1
        print(i)
        crop_box_infor(file_xml, path_img, path_output)