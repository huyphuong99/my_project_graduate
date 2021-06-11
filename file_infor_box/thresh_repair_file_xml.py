import glob
import os
import xml.etree.ElementTree as ET


def read_xml_file(path: str, name: str, thresh=.6):
    tree = ET.parse(path)
    root = tree.getroot()
    filename = root.find("filename").text
    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)
    y_min_list = []
    for obj in root.iter("object"):
        print(obj.text)
        label = obj.find("name").text
        xmin = int(obj.find("bndbox/xmin").text)
        ymin = int(obj.find("bndbox/ymin").text)
        y_min_list.append(ymin)
        xmax = int(obj.find("bndbox/xmax").text)
        ymax = int(obj.find("bndbox/ymax").text)
    y_min_list.sort()
    for obj in root.iter("object"):
        label = obj.find("name").text
        ymin = int(obj.find("bndbox/ymin").text)
        if label == name:
            if ymin == y_min_list[0]:
                pass
            else:
                label = "ISSUE BY"
        obj.find("name").text = str(label)
        with open(path, "wb") as f:
            tree.write(f)


if __name__ == "__main__":
    path = "/home/huyphuong/Desktop/material/project_tima/info_id_do_an/raw_image/raw_new_image/cropped/cropped_back/"
    name_replace = "ISSUE DATE"
    for file in glob.glob(path +"*.xml"):
        read_xml_file(file, name_replace)
        print(os.path.basename(file))
    print("done")