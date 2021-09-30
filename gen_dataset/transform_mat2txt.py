import os, sys
sys.path.append('.')
import time

import cv2

from wider_loader import WIDER
from root_path import Root_path
"""
Transfrom .mat label to .txt label format
"""
mode = 'train'
# data_path_root = '/home/data2/zhangjiaqing/widerface'
# data_path_root_mode = data_path_root + '/{}'.format(mode)
# wider face original images path
path_to_image = Root_path + "/{}/images".format(mode)

# matlab label file path
file_to_label = Root_path +"/wider_face_split/wider_face_{}.mat".format(mode)

if not os.path.exists('annotations'):
    os.makedirs('annotations')

# target annotation file path
target_file = 'annotations/wider_anno_{}.txt'.format(mode)

wider = WIDER(file_to_label, path_to_image)

line_count = 0
box_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))

        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')
print('spend time:%d'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)
