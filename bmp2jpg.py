import cv2
import glob
import os

src_path = r'C:\Users\user\PycharmProjects\yolov3\VOCdevkit\VOC2007\BMPImages'
dst_path = r'C:\Users\user\PycharmProjects\yolov3\VOCdevkit\VOC2007\JPEGImages'


file_props = glob.glob(src_path + '/*.bmp')

for file_name in file_props:
    print(file_name)
    img = cv2.imread(file_name)
    bmp_str = os.path.basename(file_name)
    save_name = dst_path + '/' + bmp_str[:-4] + '.jpg'
    cv2.imwrite(save_name, img)

