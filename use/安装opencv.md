根据[人脸检测模型 — PaddleDetection 0.1 文档](https://paddledetection.readthedocs.io/featured_model/FACE_DETECTION.html)
- #### 在FDDB数据集上评估

这一节内容测试FDDB数据集，其中安装opencv可参考
[ubuntu16.04下载opencv3.4.0并检测是否安装成功_越努力越幸运的博客-CSDN博客](https://blog.csdn.net/weixin_44741023/article/details/89604104)

注意：
因为用的是OpenCV 3.x版本，则需要修改Makefile才能编译通过，修改以下两行。

```makefile
INCS = -I/usr/local/include/opencv

LIBS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_imgcodecs
```


遇到的一些问题可参考博客：
**E: Unable to locate package libjasper-dev**
[(18条消息) E: Unable to locate package libjasper-dev的解决办法（亲测可以解决）_mango-CSDN博客](https://blog.csdn.net/qq_44830040/article/details/105961295)
**Makefile:160: recipe for target 'all' failed**
[(18条消息) linux安装opencv和Makefile:160: recipe for target 'all' failed问题解决方案_Yancy的博客-CSDN博客](https://blog.csdn.net/lyxleft/article/details/100901981)
**fatal error: dynlink_nvcuvid.h**
[解决“fatal error: dynlink_nvcuvid.h: 没有那个文件或目录#include ＜dynlink_nvcuvid.h＞“问题 - 灰信网（软件开发博客聚合）](https://www.freesion.com/article/68371362181/)

[linux下设置opencv环境变量 - linqiaozhou - 博客园 (cnblogs.com)](https://www.cnblogs.com/qiaozhoulin/p/4978055.html)

[Linux下使用FDDB 测试MTCNN人脸检测模型生成 ROC 曲线_Never__Say__No的博客-CSDN博客](https://blog.csdn.net/never__say__no/article/details/105066009)

[机器学习之分类器性能指标之ROC曲线、AUC值_zdy0_2004的专栏-CSDN博客](https://blog.csdn.net/zdy0_2004/article/details/44948511)

[hualitlc/MTCNN-on-FDDB-Dataset: Using Caffe and python to reproduce the results of MTCNN on FDDB dataset. (github.com)](https://github.com/hualitlc/MTCNN-on-FDDB-Dataset)
