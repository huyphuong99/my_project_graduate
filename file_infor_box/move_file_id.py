import glob
import shutil
import xml.etree.ElementTree as ET
import os


def move_file(path_input, path_xml_file, path_output):
    tree = ET.parse(path_xml_file)
    root = tree.getroot()
    filename = root.find("filename").text
    file_img = f"{path_input}/{filename}"
    if os.path.exists(file_img):
        for obj in root.iter("object"):
            label = obj.find("name").text
            if label == "EXPIRY":
                dest = shutil.move(file_img, path_output)
                print(dest)
                break


if __name__ == "__main__":
    path_img = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/cropped_front_id/cropped"
    path_out = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/cropped_front_id/cropped/cccd"
    path_xml = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/cropped_front_id/label_cropped/PASCAL_VOC/"
    i = 0
    for _file in glob.glob(path_xml +"/*"):
        move_file(path_img, _file, path_out)