# MTCNN_paddle
## 一、简介
本项目采用百度飞桨框架paddlepaddle复现Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks

paper：[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)

## 二、复现结果
![Results](https://github.com/icey-zhang/MTCNN_paddle/blob/main/detection_result/txtshow/DiscROC.png)
## 四、实现

### 1. 测试
#### 1）安装opencv环境和gnuplot
#### 2）数据集
数据集已挂载至aistudio项目中，如果需要本地训练可以从[这里](https://aistudio.baidu.com/aistudio/datasetdetail/109883)下载数据集，和标签文件
解压指令
```
unzip -q /home/aistudio/data/data109883/widerface.zip
unzip -q /home/aistudio/data/data109883/FDDB.zip
cd MTCNN-master 
```
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
├─MTCNN-master
```
#### 4）测试
```
python test_FDDB.py --fddb_path /home/aistudio/FDDB
```
#### 3）修改[runEvaluate.pl](https://github.com/icey-zhang/MTCNN_paddle/blob/main/evaluation/runEvaluate.pl)路径
这里由于在aistudio上装opencv没有权限，所以我在自己主机上测的
```
my $imDir = "/home/data2/zhangjiaqing/FDDB/"; 
# where the folds are
my $fddbDir = "/home/data2/zhangjiaqing/FDDB/FDDB-folds/"; 
# where the detections are
my $detDir = "/home/zhangjiaqing/zjq/MTCNN-master/detection_result/txtshow/";
```
注意：
- imDir 应该指向数据集的路径
- fddbDir 应该是指向数据集标签的路径
- detDir 是测试结果的路径

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

#### 2) 生成PNet训练数据和注释文件

```
python gen_dataset/gen_Pnet_data.py
```
```
python gen_dataset/assemble_Pnet_imglist.py
```

#### 3) 训练 PNet 模型

```
python training/pnet/train.py
```

#### 4) 生成RNet训练数据和注释文件

```
python gen_dataset/gen_Rnet_data.py
```
```
python gen_dataset/assemble_Rnet_imglist.py
```

#### 5) 训练RNet 模型
```
python training/rnet/train.py
```

#### 6) 生成ONet训练数据和注释文件

```
python gen_dataset/gen_Onet_data.py
```
```
python gen_dataset/gen_Onet_landmark.py
```
```
python gen_dataset/assemble_Onet_imglist.py
```

#### 7) 训练 ONet 模型
```
python training/onet/landmark_train.py
```
### 3. 预测一张图片
```
python predict.py --img_path /home/aistudio/MTCNN-master/img_464.jpg --base_model_path weights --detection_path detection_result/picshow/
```
预测结果保存在/MTCNN-master/detection_result/picshow路径下
![Prediction](https://github.com/icey-zhang/MTCNN_paddle/blob/main/detection_result/picshow/img_10.jpg)
