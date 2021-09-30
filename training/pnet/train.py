import sys
import os
sys.path.append(os.getcwd())
if not os.path.exists('./output'):
    os.makedirs('./output')
import paddle

from tools.dataset import FaceDataset
from nets.mtcnn import PNet
from training.pnet.trainer import PNetTrainer
import config
import os
###显卡设置
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
# Set device
use_cuda = config.USE_CUDA
# Set dataloader
# kwargs = {'num_workers': 4} if use_cuda else {}
train_data = FaceDataset(os.path.join(config.ANNO_PATH, config.PNET_TRAIN_IMGLIST_FILENAME))
val_data = FaceDataset(os.path.join(config.ANNO_PATH, config.PNET_VAL_IMGLIST_FILENAME))
dataloaders = {'train': paddle.io.DataLoader(train_data, 
                        batch_size=config.BATCH_SIZE, shuffle=True,num_workers=4),
               'val': paddle.io.DataLoader(val_data,
                        batch_size=config.BATCH_SIZE, shuffle=True,num_workers=4)
              }

# Set model
model = PNet(is_train=True)
if config.REUSE:
    model.set_state_dict(paddle.load('pretrained_weights/best_pnet.pdparams'))

# Set checkpoint
#checkpoint = CheckPoint(train_config.save_path)

# Set optimizer
scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=config.LR, milestones=config.STEPS, gamma=0.1, last_epoch=- 1, verbose=False)
optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
# Set trainer
trainer = PNetTrainer(config.EPOCHS, dataloaders, model, optimizer, scheduler)


trainer.train()
    
#checkpoint.save_model(model, index=epoch, tag=config.SAVE_PREFIX)
            
