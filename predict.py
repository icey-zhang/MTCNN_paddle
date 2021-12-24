import paddle
import os
import sys
import numpy as np


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import cv2
from MTCNN import MTCNNDetector

def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='PaddlePaddle Classification Training', add_help=add_help)

    parser.add_argument('--img_path', default='/home/aistudio/MTCNN-master/img_10.jpg', help='the image that need to be predicted')
    parser.add_argument('--detection_path', default='detection_result/picshow/', help='path where to save detection result')
    parser.add_argument('--base_model_path', dest='base_model_path', help="The save path of weights",
                    default="weights", type=str)
    args = parser.parse_args()
    return args


@paddle.no_grad()
def main(args):
    if not os.path.exists(args.detection_path):
        os.makedirs(args.detection_path)
    # define model
    p_model_path=os.path.join(args.base_model_path, 'best_pnet.pdparams')
    r_model_path=os.path.join(args.base_model_path, 'best_rnet.pdparams')#os.path.join(args.base_model_path, 'best_rnet.pth')
    o_model_path=os.path.join(args.base_model_path, 'best_onet_landmark_2.pdparams')#os.path.join(args.base_model_path, 'best_onet.pth')
    mtcnn_detector = MTCNNDetector(p_model_path,r_model_path,o_model_path,min_face_size=40,threshold=[0.7, 0.8, 0.9])


    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        bboxes,landmarks = mtcnn_detector.detect_face(image)
    except:
        bboxes = np.array([])
        landmarks = None
    print("The boxes are : ", bboxes)

    for i in range(bboxes.shape[0]):
        x0, y0, x1, y1 = bboxes[i, :4]
        width = int(x1 - x0 + 1)
        height = int(y1 - y0 + 1)
        cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)
        for j in range(5):
            x, y = int(x0 + landmarks[i, j]*width)-1, int(y0 + landmarks[i, j+5]*height)-1
            print("The {} landmark are {},{}".format(j,x,y))
            cv2.circle(image, (x, y), 2, (255, 0, 255), 2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    name= os.path.basename(args.img_path)
    save_path = args.detection_path + name
    cv2.imwrite(save_path, image)


if __name__ == "__main__":
    args = get_args()
    main(args)
