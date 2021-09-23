import time
from recognition.detect_frame_and_recognition import main_project

if __name__ == "__main__":
    st = time.time()
    path = "/home/huyphuong99/Desktop/material/test/id/test_cmt_2.jpg"
    ratio = 1 / 20.0
    main_project(path, ratio)
    print(time.time() - st)
