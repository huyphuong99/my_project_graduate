import glob
import os
import pandas as pd


def label_list():
    for file in glob.glob(path_img):
        file_name = os.path.basename(file)
        try:
            if file_name in data_csv:
                name = file_name.split("_")[-1].replace(".jpg", "")
                x = len(name)-1
                if name[x] == ",":
                    name = name[:len(name)-1]
                idx = data_csv.index(file_name)
                data_fr['label'][idx] = name
        except Exception as e:
            print(f"{e}:{file_name}")



if __name__ == "__main__":
    list_data = ["issue_date", "religion"]
    folders = "back"
    # list_data = ["BIRTHDAY", "ISSUEDATE"]
    # folders = "DAY"
    # list_data = ["NAME"]
    # folders =""
    total = 0
    for fd in list_data:
        path_img = f"/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/output/{folders}/{fd}/*"
        path_csv = f"/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/output/{folders}/{fd}.csv"
        data_fr = pd.read_csv(path_csv, dtype='str')
        data_csv = [f for f in data_fr["filename"]]
        label_list()
        total += len(data_fr)
        print(f'File {fd}: {data_fr}')
        data_fr.to_csv(f'/home/huyphuong99/PycharmProjects/project_totnghiep/temp_compress/output/{folders}/{fd}.csv')
    print(f'Total file: {total}')
