import paddle

from loss import Loss
from tools.average_meter import AverageMeter
from accuracy import compute_accuracy
from utils import setup_logger
import logging
import time
setup_logger('base', 'output', 'onet-train', level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')

class ONetTrainer(object):
    
    def __init__(self, epochs, dataloaders, model, optimizer, scheduler):
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfn = Loss()
        
        # save best model
        self.best_val_loss = 100
    
    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch, 'train')
            self.train_epoch(epoch, 'val')

        
    def train_epoch(self, epoch, phase):
        cls_loss_ = AverageMeter()
        bbox_loss_ = AverageMeter()
        total_loss_ = AverageMeter()
        accuracy_ = AverageMeter()
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        #training log
        train_reader_cost = 0.0
        train_run_cost = 0.0
        total_samples = 0
        # acc = 0.0
        reader_start = time.time()
        batch_past = 0
        ###############

        for batch_idx, sample in enumerate(self.dataloaders[phase]):
            #training log
            train_reader_cost += time.time() - reader_start
            train_start = time.time()
            ###############
            data = sample['input_img']
            gt_cls = sample['cls_target']
            gt_bbox = sample['bbox_target']
            gt_bbox = paddle.to_tensor(gt_bbox,dtype='float32')
            
            self.optimizer.clear_grad()
            with paddle.set_grad_enabled(phase == 'train'):
                pred_cls, pred_bbox = self.model(data)
                
                # compute the cls loss and bbox loss and weighted them together
                cls_loss = self.lossfn.cls_loss(gt_cls, pred_cls)
                bbox_loss = self.lossfn.box_loss(gt_cls, gt_bbox, pred_bbox)
                total_loss = cls_loss + 10*bbox_loss
                
                # compute clssification accuracy
                accuracy = compute_accuracy(pred_cls, gt_cls)

                if phase == 'train':
                    total_loss.backward()
                    self.optimizer.step()

            #training log
            train_run_cost += time.time() - train_start
            total_samples += data.shape[0]
            batch_past += 1
            ###############

            cls_loss_.update(cls_loss, data.shape[0])
            bbox_loss_.update(bbox_loss, data.shape[0])
            total_loss_.update(total_loss, data.shape[0])
            accuracy_.update(accuracy, data.shape[0])
            
            if batch_idx % 40 == 0:
                logger.info('{} Epoch: {} [{:08d}/{:08d} ({:02.0f}%)]\tLoss: {:.6f} cls Loss: {:.6f} offset Loss:{:.6f}\tAccuracy: {:.6f}\tavg_reader_cost: {:.5f} sec, avg_batch_cost: {:.5f} sec, avg_samples: {}, avg_ips: {:.5f} images/sec.'.format( #LR:{:.7f}
                    phase, epoch, batch_idx * len(data), len(self.dataloaders[phase].dataset),
                    100. * batch_idx / len(self.dataloaders[phase].dataset), total_loss.item(), cls_loss.item(), bbox_loss.item(), accuracy.item(), train_reader_cost / batch_past,
                    (train_reader_cost + train_run_cost) / batch_past,
                    total_samples / batch_past,
                    total_samples / (train_reader_cost + train_run_cost)))

                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
                # acc = 0.0
                batch_past = 0


            reader_start = time.time()
            ###############
        
        # if epoch % 10 == 0:
        #     paddle.save(self.model.state_dict(), './pretrained_weights_onet/{}_onet.pdparams'.format(epoch))        

        if phase == 'train':
            self.scheduler.step()
        
        logger.info("{} epoch Loss: {:.6f} cls Loss: {:.6f} bbox Loss: {:.6f} Accuracy: {:.6f}".format(
            phase, total_loss_.avg.item(), cls_loss_.avg.item(), bbox_loss_.avg.item(), accuracy_.avg.item()))
        
        if phase == 'val' and total_loss_.avg < self.best_val_loss:
            self.best_val_loss = total_loss_.avg
            paddle.save(self.model.state_dict(), './weights/best_onet.pdparams')
        
        return cls_loss_.avg, bbox_loss_.avg, total_loss_.avg, accuracy_.avg
    
