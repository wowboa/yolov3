import sys
import argparse
from yolo_autolabel import YOLO, detect_video
from PIL import Image
import os
import glob
from xml.dom.minidom import Document
import numpy as np
import xml.etree.ElementTree as et

def get_xml_info(file):
    tree = et.parse(file)
    root = tree.getroot()
    boxes = []
    class_name = []

    for Object in root.findall('object'):
        name = Object.find('name').text
        bndbox = Object.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        boxes.append([int(xmin),int(ymin),int(xmax),int(ymax)])
        class_name.append(name)
    return boxes, class_name

def IOU(x1,y1,X1,Y1, x2,y2,X2,Y2):
    xx = max(x1,x2)
    XX = min(X1,X2)
    yy = max(y1,y2)
    YY = min(Y1,Y2)
    m = max(0., XX-xx)
    n = max(0., YY-yy)
    Jiao = m*n
    Bing = (X1-x1)*(Y1-y1)+(X2-x2)*(Y2-y2)-Jiao
    return Jiao/Bing
def get_iou(t_box, box):
    # box [ymin, xmin, ymax, xmax]
    xx = max(t_box[0], box[1])
    XX = min(t_box[2], box[3])
    yy = max(t_box[1], box[0])
    YY = min(t_box[3], box[2])
    m = max(0., XX - xx)
    n = max(0., YY - yy)
    Jiao = m * n
    Bing = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1]) + (box[2] - box[0]) * (box[3] - box[1]) - Jiao
    return Jiao / Bing

def get_metrics(yolo):
    img_path = "/home/lzb/remotedata/yolov3/VOCdevkit/VOC2007/SegmentationObjectxxxxx/*.jpg"
    ann_path = "/home/lzb/remotedata/NG_pics/keras-yolo3/VOCdevkit/VOC2007/Annotations"
    TP = 0
    P  = 0
    T  = 0

    for jpgfile in glob.glob(img_path):
        name = jpgfile.split('/')[-1].split('.')[0]
        img = Image.open(jpgfile)
        boxes, classes, imagesize, class_name = yolo.detect_image(img)
        # box [ymin, xmin, ymax, xmax]
        #print('boxes:',boxes)
        #print('class_name:', class_name)
        true_boxes, true_class_name = get_xml_info(ann_path+'/'+name+'.xml')
        #print('true_boxes:', true_boxes)
        #print('true_class_name:', true_class_name)
        P += len(boxes)
        T += len(true_boxes)
        for i in range(len(true_boxes)):
            for j in range(len(boxes)):
                if get_iou(true_boxes[i], boxes[j]) >= 0.1 and true_class_name[i]==class_name[classes[j]]:
                    TP += 1
                    break
    precision = TP/P
    recall    = TP/T
    print('Precision:', precision)
    print('Recall:', recall)


    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        get_metrics(YOLO(**vars(FLAGS)))
        #print('Start write info to xml...')
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")