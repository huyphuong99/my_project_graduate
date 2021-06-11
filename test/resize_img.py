import cv2

def resize_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512,512))
    cv2.imwrite(f"{path}", img)
path = "/home/huyphuong/Desktop/material/test/img1.jpg"
# resize_img(path)
