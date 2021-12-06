# MTCNN_paddle
## 一、简介
本项目采用百度飞桨框架paddlepaddle复现Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

paper：[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)

## 二、复现结果
![Results](https://github.com/icey-zhang/MTCNN_paddle/blob/main/detection_result/txtshow/DiscROC.png)
## 四、实现

### 1. 测试
#### 1）安装opencv环境和gnuplot
[参考](https://github.com/icey-zhang/MTCNN_paddle/blob/main/use/%E5%AE%89%E8%A3%85opencv.md)

#### 2）数据集
[下载链接aistudio](https://aistudio.baidu.com/aistudio/datasetdetail/110657)

其中train文件放到widerface目录下，命名为FacePoint

[RetinaFace-WIDER FACE](https://aistudio.baidu.com/aistudio/datasetdetail/104236)

[FDDB人脸检测数据集](https://aistudio.baidu.com/aistudio/datasetdetail/37474)

```
├─FDDB
   ├─2002
   ├─2003
   ├─FDDB-folds
├─widerface
   ├─FacePoint
   ├─wider_face_split
   ├─train
   ├─val
   ├─test
├─MTCNN-master
```
#### 4）测试
```
python test_FDDB.py --fddb_path /home/aistudio/FDDB
```
#### 3）修改[runEvaluate.pl](https://github.com/icey-zhang/MTCNN_paddle/blob/main/evaluation/runEvaluate.pl)路径
```
my $imDir = "/home/data2/zhangjiaqing/FDDB/"; 
# where the folds are
my $fddbDir = "/home/data2/zhangjiaqing/FDDB/FDDB-folds/"; 
# where the detections are
my $detDir = "/home/zhangjiaqing/zjq/MTCNN-master/detection_result/txtshow/";
```
#### 4）评估
```
cd evaluation
perl runEvaluate.pl
```
### 2. 训练

以下各节以分步方式描述数据准备和网络训练

#### 1) 准备 Wider_Face 注释文件
修改目录
原始的宽脸注释文件是matlab格式。让我们将它们转换为.txt文件。

```
python gen_dataset/transform_mat2txt.py
```

修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件


#### 2) 生成PNet训练数据和注释文件

```
python gen_dataset/gen_Pnet_train_data.py
```
修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件

```
python gen_dataset/assemble_Pnet_imglist.py
```
修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件

#### 3) 训练 PNet 模型

```
python training/pnet/train.py
```
#### 4) 生成RNet训练数据和注释文件

```
python gen_dataset/gen_Rnet_train_data.py
```
修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件
```
python gen_dataset/assemble_Rnet_imglist.py
```
修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件
#### 5) 训练RNet 模型
```
python training/rnet/train.py
```
#### 6) 生成ONet训练数据和注释文件

```
python gen_dataset/gen_Onet_train_data.py
```

修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件
```
python gen_dataset/gen_Onet_landmark.py
```
修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件
```
python gen_dataset/assemble_Onet_imglist.py
```

修改mode=‘val’或者mode=‘train’再重复一边生成验证/训练文件

#### 7) 训练 ONet 模型
```
python training/onet/landmark_train.py
```



