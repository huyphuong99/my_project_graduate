from typing import List, Dict

import cv2
import numpy as np
import requests
from recognition import detect_frame_and_recognition

BASE_URL = 'http://172.16.1.11:5000'
VERSION = "phuongdz_ver2_180621"


def predict(img: np.ndarray, name) -> Dict:
    # ================================== Edit code here ========================
    # input: image
    # output: dictionary with key follow example:
    dict_infor = detect_frame_and_recognition.main_project(img, name)
    id, name, birthday, address, domicile, expiry, gender, country, ethnicity, issue_by, issue_date, religion = \
        None, None, None, None, None, None, None, None, None, None, None, None
    if  dict_infor is not None and "ID" in dict_infor.keys():
        id = dict_infor['ID']
        name = dict_infor['NAME']
        if 'BIRTHDAY' in dict_infor.keys():
            birthday = dict_infor['BIRTHDAY']
        address = dict_infor['ADDRESS']
        domicile = dict_infor['DOMICILE']
        if "EXPIRY" in dict_infor.keys():
            expiry = dict_infor['EXPIRY']
        if "SEX" in dict_infor.keys():
            gender = dict_infor['SEX']
        if "COUNTRY" in dict_infor.keys():
            country = dict_infor['COUNTRY']
        if "ETHNICITY" in dict_infor.keys():
            ethnicity = dict_infor["ETHNICITY"]
    elif dict_infor is not None:
        if 'ISSUE BY' in dict_infor.keys():
            issue_by = dict_infor['ISSUE BY']
        if 'ISSUE DATE' in dict_infor.keys():
            issue_date = dict_infor['ISSUE DATE']
        if "ETHNICITY" in dict_infor.keys():
            ethnicity = dict_infor['ETHNICITY']
        if "RELIGION" in dict_infor.keys():
            religion = dict_infor['RELIGION']
    return {
        "id_number": id,
        "address": address,
        "domicile": domicile,
        "birthday": birthday,
        "ethnicity": ethnicity,
        "expiry": expiry,
        "gender": gender,
        "issue_by": issue_by,
        "issue_date": issue_date,
        "name": name,
        "religion": religion,
        "country": country
    }


def get_list_files() -> List[str]:
    res = requests.post(f'{BASE_URL}/api/v1/images')
    return res.json()


def get_image(filename: str) -> np.ndarray:
    res = requests.get(f'{BASE_URL}/api/v1/image/{filename}')
    arr = np.asarray(bytearray(res.content), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img


def submit_result(filename: str, result: Dict) -> bool:
    res = requests.put(f'{BASE_URL}/api/v1/ocr/id/predict/{VERSION}/{filename}', json=result)
    return res.status_code == 200 and res.json()['status_code'] == 200


def run():
    filenames = get_list_files()
    print(f"Total: {len(filenames)} images.")
    for i, filename in enumerate(filenames):
        print(f'Process {i + 1}/{len(filenames)}: {filename}', end="\t\t\t")
        img = get_image(filename)
        res = predict(img,filename)
        print(res)
        submitted = submit_result(filename, res)
        print(submitted)


if __name__ == '__main__':
    run()
