import sys
sys.path.append('./')
import os
import argparse

import paddle

from tools.dataset import FaceDataset
from nets.mtcnn import ONet
from training.onet.landmark_trainer import ONetTrainer
import config

# Set device
use_cuda = config.USE_CUDA

# Set dataloader
kwargs = {'num_workers': 4} if use_cuda else {}
train_data = FaceDataset(os.path.join(config.ANNO_PATH, config.ONET_TRAIN_IMGLIST_FILENAME))
val_data = FaceDataset(os.path.join(config.ANNO_PATH, config.ONET_VAL_IMGLIST_FILENAME))
dataloaders = {'train': paddle.io.DataLoader(train_data, 
                        batch_size=config.BATCH_SIZE, shuffle=True, **kwargs),
               'val': paddle.io.DataLoader(val_data,
                        batch_size=config.BATCH_SIZE, shuffle=True, **kwargs)
              }

# Set model
model = ONet(is_train=True, train_landmarks=True)
if config.REUSE:
    model.load_state_dict(paddle.load('pretrained_weights/mtcnn/best_onet.pdparams'))
print(model)

# Set checkpoint
#checkpoint = CheckPoint(train_config.save_path)

# Set optimizer
scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config.LR, milestones=config.STEPS, gamma=0.1, last_epoch=- 1, verbose=False)
optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())


# Set trainer
trainer = ONetTrainer(config.EPOCHS, dataloaders, model, optimizer, scheduler)

trainer.train()
    
#checkpoint.save_model(model, index=epoch, tag=config.SAVE_PREFIX)
            
