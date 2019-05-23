import cv2
import glob
import os
import xml.etree.cElementTree as et

'''
调用xml库中的相关方法可以快速实现xml文件中的信息的读取
此处读取的xml文件的格式为VOC中的数据标注格式，以下脚本展示了
读取所有的bounding box的坐标的方法，相应的可以按照以下方式
读取其他的信息。
'''

tree = et.parse("R1142.xml")
root = tree.getroot()

filename = root.find('filename').text
print(filename)

for Object in root.findall('object'):
    name = Object.find('name').text
    print(name)
    bndbox = Object.find('bndbox')
    xmin = bndbox.find('xmin').text
    ymin = bndbox.find('ymin').text
    xmax = bndbox.find('xmax').text
    ymax = bndbox.find('ymax').text
    print(xmin, ymin, xmax, ymax)


dir_xml = r'C:\Users\user\PycharmProjects\yolov3\VOC2007\Annotations'
dir_bmp = r'C:\Users\user\PycharmProjects\yolov3\VOC2007\JPEGImages'

file_xml = glob.glob(dir_xml + '/*.xml')

for i in range(10):
    filename = file_xml[i]
    basename = os.path.basename(filename)
    bmp_name = basename[:-3] + 'bmp'
    img = cv2.imread(dir_bmp + '/' + bmp_name)

    tree = et.parse(filename)
    root = tree.getroot()

    filename = root.find('filename').text
    print(filename)

    for Object in root.findall('object'):
        name = Object.find('name').text
        print(name)
        bndbox = Object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        print(xmin, ymin, xmax, ymax)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)

    cv2.imshow('def', img)
    cv2.waitKey(0)
