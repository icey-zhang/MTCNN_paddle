import sys
# sys.path.append('..')
import os
import os.path
# from torch import device
import paddle
sys.path.append(os.getcwd())
import cv2
from tools.utils import *
from MTCNN import MTCNNDetector
import argparse

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MTCNN Demo')
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default="20", type=int)
    parser.add_argument('--base_model_path', dest='base_model_path', help="The save path of weights",
                        default="weights", type=str)
    parser.add_argument('--fddb_path', dest='fddb_path', help="The path of datasets",
                        default="/home/aistudio/FDDB", type=str)
    
    detection_path = 'detection_result'
    if not os.path.exists(detection_path):
        os.makedirs(detection_path + '/picshow')
        os.makedirs(detection_path + '/txtshow')
    args = parser.parse_args()
    
    gt_box_dict = get_gt_boxes(os.path.join(args.fddb_path, 'FDDB-folds'))

    p_model_path=os.path.join(args.base_model_path, 'best_pnet.pdparams')
    r_model_path=os.path.join(args.base_model_path, 'best_rnet.pdparams')#os.path.join(args.base_model_path, 'best_rnet.pth')
    o_model_path=os.path.join(args.base_model_path, 'best_onet_landmark_2.pdparams')#os.path.join(args.base_model_path, 'best_onet.pth')
    mtcnn_detector = MTCNNDetector(p_model_path,r_model_path,o_model_path,threshold=[0.1, 0.1, 0.1]
    )
    running_correct = 0.0
    running_gt = 0.0
    for fold_index in range(1, 11):
        t=0
        with open(os.path.join(args.fddb_path, 'FDDB-folds', 'FDDB-fold-{:02d}.txt'.format(fold_index)), 'r') as f, \
                open("detection_result/txtshow/fold-{:02d}-out.txt".format(fold_index), "w") as fw:
            lines = f.readlines()
            for line in lines:
                image_path = line.strip() + '.jpg'
                # print(image_path)
                fw.write(line.strip() + "\n")
                gt_label = gt_box_dict[fold_index][line.strip().replace('/','_')]
                image = cv2.imread(os.path.join(args.fddb_path, image_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # bboxess,landmarks = mtcnn_detector.detect_face(image)
                try:
                    bboxes,landmarks = mtcnn_detector.detect_face(image)
                    t=t+1
                except:
                    bboxes = np.array([])
                    landmarks = None
                dets = bboxes
                # print(dets)
                fw.write(str(len(dets)) + "\n")
                for b in dets:
                    x = str(int(b[0]))
                    y = str(int(b[1]))
                    w = str(int(b[2]) - int(b[0]))
                    h = str(int(b[3]) - int(b[1]))
                    confidence = str(b[4])

                    fw.write("{} {} {} {} {}\n".format(x, y, w, h, confidence))
                    # fw.flush()

                for i in range(bboxes.shape[0]):
                    x0, y0, x1, y1 = bboxes[i, :4]
                    cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 1)

                if landmarks is not None:
                    for i in range(landmarks.shape[0]):
                        landmark = landmarks[i, :]
                        landmark = landmark.reshape(2, 5).T
                        for j in range(5):
                            cv2.circle(image, (int(landmark[j, 0]), int(landmark[j, 1])), 2, (0, 255, 255), 1)
                
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                name= os.path.basename(image_path)
                save_path = detection_path + '/picshow/'+name
                cv2.imwrite(save_path, image)
        print(t)
        f.close()
        fw.close()
    
    # filedir = 'detection_result/txtshow'  # 填入要合并的文件夹名字
    # filenames = os.listdir(filedir)  # 获取文件夹内每个文件的名字
    # f = open('FDDB/FDDB_results.txt', 'w')  # 以写的方式打开文件，没有则创建

    # # 对每个文件进行遍历
    # for filename in filenames:
    #     filepath = filedir + '/' + filename  # 将文件夹路径和文件名字合并
    #     for line in open(filepath):  # 循环遍历对每一个文件内的数据
    #         f.writelines(line)  # 将数据每次按行写入f打开的文件中

    # f.close()  # 关闭

