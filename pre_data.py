import glob
import os
import shutil


srcdir = r'D:\dong\wafer-data\data_bmp'
seldir = r'C:\Users\user\PycharmProjects\yolov3\VOC2007\Annotations'
dstdir = r'C:\Users\user\PycharmProjects\yolov3\VOC2007\JPEGImages'

file_props = glob.glob(seldir + '/*.xml')

for filename in file_props:
    basename = os.path.basename(filename)
    bmp_name = basename[:-3] + 'bmp'
    srcfile = srcdir + '/' + bmp_name
    dstfile = dstdir + '/' + bmp_name
    print(srcfile)
    shutil.copy(srcfile, dstfile)
