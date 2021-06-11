import xml.etree.ElementTree as ET
from load_model_detect_frame import main_detection

def create_xml(dict_image, path_output):
    data_annotation = ET.ElementTree("annotation")
    folder = ET.SubElement("folder")
    file_name = ET.SubElement("filename")
    path = ET.SubElement("path")
    source = ET.SubElement("source")
    data_base = ET.SubElement(source, "database")
    size = ET.SubElement("size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size,"depth")
    segmented = ET.SubElement("segmented")
    for i in dict_image:
        object = ET.SubElement("object")
        name = ET.SubElement(object, "name")
        pose = ET.SubElement(object, "pose")
        truncated = ET.SubElement(object, "truncated")
        difficult = ET.SubElement(object, "difficult")
        bndbox = ET.SubElement(object, "bndbox")
        xmin = ET.SubElement(object, "xmin")
        ymin = ET.SubElement(object, "ymin")
        xmax = ET.SubElement(object, "xmax")
        ymax = ET.SubElement(object, "ymax")

    b_xml = ET.tostring(data_annotation)
    with open(path_output, "wb") as f:
        f.write(b_xml)

if __name__ == "__main__":
    path = ["/home/huyphuong/Desktop/material/project_tima/info_id_do_an/raw_image/raw_new_image/new_cccd/front/16347_10030626_1534087.jpg"]
    dict_img = main_detection(path[0])
    print(dict_img)
    # create_xml(dict_img)
