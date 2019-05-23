import sys
import argparse
from yolo_autolabel import YOLO, detect_video
from PIL import Image
import os
import glob
from xml.dom.minidom import Document
import numpy as np

def make_xml(boxes, classes, imagesize, name, class_name):
    doc = Document()  # 创建DOM文档对象
    DOCUMENT = doc.createElement('annotation')  # 创建根元素
    doc.appendChild(DOCUMENT)

    #folder
    folder = doc.createElement('folder')
    DOCUMENT.appendChild(folder)
    folder_txt = doc.createTextNode('0')
    folder.appendChild(folder_txt)

    filename = doc.createElement('filename')
    DOCUMENT.appendChild(filename)
    filename_txt = doc.createTextNode(name+'.jpg')
    filename.appendChild(filename_txt)

    path = doc.createElement('path')
    DOCUMENT.appendChild(path)
    path_txt = doc.createTextNode(name + '.jpg')
    path.appendChild(path_txt)

    #source
    source = doc.createElement('source')
    DOCUMENT.appendChild(source)
    database = doc.createElement('database')
    source.appendChild(database)
    database_txt = doc.createTextNode('Unknown')
    database.appendChild(database_txt)

    #size
    size = doc.createElement('size')
    DOCUMENT.appendChild(size)
    width = doc.createElement('width')
    size.appendChild(width)
    width_txt = doc.createTextNode(str(imagesize[0]))
    width.appendChild(width_txt)
    heigth = doc.createElement('height')
    size.appendChild(heigth)
    heigth_txt = doc.createTextNode(str(imagesize[1]))
    heigth.appendChild(heigth_txt)
    depth = doc.createElement('depth')
    size.appendChild(depth)
    depth_txt = doc.createTextNode('3')
    depth.appendChild(depth_txt)

    #segmented
    segmented = doc.createElement('segmented')
    DOCUMENT.appendChild(segmented)
    segmented_txt = doc.createTextNode('0')
    segmented.appendChild(segmented_txt)

    for i, c in reversed(list(enumerate(classes))):
        object = doc.createElement('object')
        DOCUMENT.appendChild(object)
        #name
        classname = doc.createElement('name')
        object.appendChild(classname)
        classname_txt = doc.createTextNode(class_name[c])
        classname.appendChild(classname_txt)
        #pose
        pose = doc.createElement('pose')
        object.appendChild(pose)
        pose_txt = doc.createTextNode('Unspecified')
        pose.appendChild(pose_txt)
        # truncated
        truncated = doc.createElement('truncated')
        object.appendChild(truncated)
        truncated_txt = doc.createTextNode('0')
        truncated.appendChild(truncated_txt)
        # difficult
        difficult = doc.createElement('difficult')
        object.appendChild(difficult)
        difficult_txt = doc.createTextNode('0')
        difficult.appendChild(difficult_txt)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        box = boxes[i]
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(imagesize[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(imagesize[0], np.floor(right + 0.5).astype('int32'))
        xmin = doc.createElement('xmin')
        bndbox.appendChild(xmin)
        xmin_txt = doc.createTextNode(str(left))
        xmin.appendChild(xmin_txt)
        ymin = doc.createElement('ymin')
        bndbox.appendChild(ymin)
        ymin_txt = doc.createTextNode(str(top))
        ymin.appendChild(ymin_txt)
        xmax = doc.createElement('xmax')
        bndbox.appendChild(xmax)
        xmax_txt = doc.createTextNode(str(right))
        xmax.appendChild(xmax_txt)
        ymax = doc.createElement('ymax')
        bndbox.appendChild(ymax)
        ymax_txt = doc.createTextNode(str(bottom))
        ymax.appendChild(ymax_txt)
    return doc



def detect_img_write_xml(yolo):
    path = "/home/lzb/remotedata/yolov3/VOCdevkit/VOC2007/SegmentationObject/*.jpg"
    outdir = "/home/lzb/remotedata/yolov3/VOCdevkit/VOC2007/SegmentationClass"
    for jpgfile in glob.glob(path):
        name = jpgfile.split('/')[-1].split('.')[0]
        img = Image.open(jpgfile)
        boxes, classes, imagesize, class_name = yolo.detect_image(img)
        doc = make_xml(boxes, classes, imagesize, name, class_name)
        with open(outdir+'/'+name+'.xml', 'w') as f:
            doc.writexml(f,indent = '\t',newl = '\n', addindent = '\t',encoding='utf-8')

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
        detect_img_write_xml(YOLO(**vars(FLAGS)))
        print('Start write info to xml...')
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")