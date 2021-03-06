"""
Generate positive, negative, positive images whose size are 24*24 from Pnet.
"""
import sys
sys.path.append('.')
import cv2
import os
import numpy as np
from tools.utils import*
from MTCNN import MTCNNDetector
import config
from root_path import Root_path


for mode in ['train','val']:
    # mode = 'train'
    image_size = config.RNET_SIZE
    prefix = ''
    anno_file = "./annotations/wider_anno_{}.txt".format(mode)

    im_dir = Root_path + "/{}/images".format(mode) #val?????

    pos_save_dir = Root_path + "/{}/{}/positive".format(mode,image_size)
    part_save_dir = Root_path + "/{}/{}/part".format(mode,image_size)
    neg_save_dir = Root_path + "/{}/{}/negative".format(mode,image_size)

    if not os.path.exists(pos_save_dir):
        os.makedirs(pos_save_dir)
    if not os.path.exists(part_save_dir):
        os.makedirs(part_save_dir)
    if not os.path.exists(neg_save_dir):
        os.makedirs(neg_save_dir)

    # store labels of positive, negative, part images
    f1 = open(os.path.join('annotations', 'pos_24_{}.txt'.format(mode)), 'w')
    f2 = open(os.path.join('annotations', 'neg_24_{}.txt'.format(mode)), 'w')
    f3 = open(os.path.join('annotations', 'part_24_{}.txt'.format(mode)), 'w')

    # anno_file: store labels of the wider face training data
    with open(anno_file, 'r') as f:
        annotations = f.readlines()
    num = len(annotations)
    print("%d pics in total" % num)


    p_idx = 0 # positive
    n_idx = 0 # negative
    d_idx = 0 # dont care
    idx = 0

    # create MTCNN Detector
    mtcnn_detector = MTCNNDetector(p_model_path='./weights/best_pnet.pdparams')

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_path = os.path.join(prefix, annotation[0])
        # print(im_path)
        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        # anno form is x1, y1, w, h, convert to x1, y1, x2, y2
        boxes[:,2] += boxes[:,0] - 1
        boxes[:,3] += boxes[:,1] - 1

        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = mtcnn_detector.detect_face(image)
        
        # bboxes, landmarks = create_mtcnn_net(image, 12, device, p_model_path='../train/pnet_Weights')
        if bboxes.shape[0] == 0:
            continue
        
        dets = np.round(bboxes[:, 0:4])


        img = cv2.imread(im_path)
        idx += 1

        height, width, channel = img.shape

        for box in dets:
            x_left, y_top, x_right, y_bottom = box[0:4].astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            # ignore box that is too small or beyond image border
            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            # compute intersection over union(IoU) between current box and all gt boxes
            Iou = IoU(box, boxes)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size),
                                    interpolation=cv2.INTER_LINEAR)

            # save negative images and write label
            if np.max(Iou) < 0.2 and n_idx < 3.0*p_idx+1:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find gt_box with the highest iou
                idx_Iou = np.argmax(Iou)
                assigned_gt = boxes[idx_Iou]
                x1, y1, x2, y2 = assigned_gt

                # compute bbox reg label
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                # save positive and part-face images and write labels
                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4 and d_idx < 1.0*p_idx + 1:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1

        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

        #if idx == 20:
        #     break

    f1.close()
    f2.close()
    f3.close()
