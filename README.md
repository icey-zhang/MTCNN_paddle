# Repo for training, optimizing and deploying MTCNN for mobile devices

## Prepare dataset
1. Download [WIDER FACE]() dataset and put them under `./data` directory.
2. Transform matlab format label train and val format file into text format.
   RUN `pyhon gen_dataset/transform_mat2txt.py`
   Change the mode variable in `transform_mat2txt.py` to `val` to generate val data label.
   
## Train MTCNN
### Train Pnet
1. generate pnet training data:
    RUN `python gen_dataset/gen_Pnet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Pnet:
    RUN `python training/pnet/train.py` to train your model.
3. Save weights:
    We use the validation dataset to help us with choose the best pnet model. The weights are saved in `pretrained_weights/mtcnn/best_pnet.pth`

### Train Rnet
After we trained Pnet, we can use Pnet to generate data for training Rnet.
1. generate Rnet training data:
    RUN `python gen_dataset/gen_Rnet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Rnet:
    RUN `python training/rnet/train.py` to train your model.
3. 3. Save weights:
    We use the validation dataset to help us with choose the best rnet model. The weights are saved in `pretrained_weights/mtcnn/best_rnet.pth`
    
### Train Onet
After we trained Pnet and Rnet, we can use Pnet and Rnet to generate data for training Onet.
1. generate Onet training data:
    RUN `python gen_dataset/gen_Onet_data.py`
    Change the mode variable in `gen_Pnet_data.py` to `val` to generate val data label.
2. Training Onet:
    RUN `python training/Onet/train.py` to train your model.
3. 3. Save weights:
    We use the validation dataset to help us with choose the best onet model. The weights are saved in `pretrained_weights/mtcnn/best_onet.pth`
    
### Results

|  WIDER FACE |  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.156 | 0.120| 0.129 |
| offset loss |  0.01  | 0.01 | 0.0063|
|   cls acc   |  0.944 | 0.962| 0.956 |

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.05  | 0.09 | 0.104 |
| offset loss | 0.0047 | 0.011 | 0.0057|
|   cls acc   |  0.983 | 0.971 | 0.970 |

## Optimize MTCNN
### Lighter MTCNN
By combine shufflenet structure and mobilenet structure we can design light weight Pnet, Rnet, and Onet. In this way can can optimize the size of the model and at the same time decrease the inference speeed.

### Larger Pnet
According to my observation, small pnet brings many false positives which becomes a burden or rnet and onet. By increase the Pnet size, there will be less false positives and improve the overall efficiency.

### Prune MTCNN

Model Prunning is a better strategy than design mobile cnn for such small networks as Pnet, Rnet, and Onet. By iteratively pruning MTCNN models, we can decrease and model size and improve inference speed at the same time. 

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.091  | 0.1223 | 0.1055 |
| offset loss | 0.0055 | 0.0116 | 0.0062 |
|   cls acc   |  0.970 | 0.958 | 0.959 |

Inference speed benchmark using ncnn inference framework, we can seen from the chart below that the inference speed has been increased by 2-3 times.

```
       pnet  min =   27.31  max =   28.31  avg =   27.62
       rnet  min =    0.50  max =    0.62  avg =    0.58
       onet  min =    3.14  max =    3.82  avg =    3.25
pruned_pnet  min =    6.76  max =    7.13  avg =    6.89
pruned_rnet  min =    0.21  max =    0.22  avg =    0.21
pruned_onet  min =    1.16  max =    1.52  avg =    1.27
```
				  
We could also treat prunning process as a *NAS(network architecture search)* procesure. After we obtained the model, we could train it from zero. And I achieved better accuracy using this method on pruned model above.

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  | 0.0083  | 0.1038 | 0.0923 |
| offset loss | 0.00588 | 0.012 | 0.00588 |
|   cls acc   | 0.9718 | 0.965 | 0.9706 |

### Quantization Aware Training

By using quantization aware training library [brevitas](https://github.com/Xilinx/brevitas), I managed to achieve 96.2% accuracy on Pnet which is 2% lower than the original version, but the model size if 4x smaller and the inference speed is to be estimated.

However, when training Rnet and Onet, OOM errors occured. I will figure out why in the future.

| PRIVATE DATA|  Pnet  |  Rnet |  Onet |
| :---------: |:------:|:-----:|:-----:|
|   cls loss  |  0.107  | - | - |
| offset loss | 0.0080 | - | - |
|   cls acc   |  0.962 | - | - |


### Knowledge Distillation

## Deploy MTCNN


## Todo
- [ ] Data Augmentation to avoid overfitting
- [ ] Use L1 Smooth loss or WingLoss for bbox and landmarks localization


## References
1. https://github.com/xuexingyu24/MTCNN_Tutorial
2. https://github.com/xuexingyu24/Pruning_MTCNN_MobileFaceNet_Using_Pytorch

