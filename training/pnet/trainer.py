import paddle
from loss import Loss
from tools.average_meter import AverageMeter
from utils import setup_logger
import logging
setup_logger('base', 'output', 'pnet-train', level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
from accuracy import compute_accuracy
class PNetTrainer(object):
    
    def __init__(self, epochs, dataloaders, model, optimizer, scheduler):
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lossfn = Loss()
        
        # save best model
        self.best_val_loss = 100
        self.start_epoch = 64
        self.model.set_state_dict(paddle.load('./pretrained_weights/63_pnet.pdparams'))
    
    def train(self):
        paddle.load
        for epoch in range(self.start_epoch,self.epochs):
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

        for batch_idx, sample in enumerate(self.dataloaders[phase]):
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
                total_loss = cls_loss + 5*bbox_loss
                
                # compute clssification accuracy
                accuracy = compute_accuracy(pred_cls, gt_cls)

                if phase == 'train':
                    total_loss.backward()
                    self.optimizer.step()

            cls_loss_.update(cls_loss, data.shape[0])
            bbox_loss_.update(bbox_loss, data.shape[0])
            total_loss_.update(total_loss, data.shape[0])
            accuracy_.update(accuracy, data.shape[0])
            
            if batch_idx % 40 == 0:
                logger.info('{} Epoch: {} [{:08d}/{:08d} ({:02.0f}%)]\tLoss: {:.6f} cls Loss: {:.6f} offset Loss:{:.6f}\tAccuracy: {:.6f}'.format( #LR:{:.7f}
                    phase, epoch, batch_idx * len(data), len(self.dataloaders[phase].dataset),
                    100. * batch_idx / len(self.dataloaders[phase]), total_loss.item(), cls_loss.item(), bbox_loss.item(), accuracy.item())) #, self.optimizer.param_groups[0]['lr']
        # if epoch % 10 == 0:
        #     paddle.save(self.model.state_dict(), './pretrained_weights_pnet/{}_pnet.pdparams'.format(epoch))
        
        if phase == 'train':
            self.scheduler.step()
        
        logger.info("{} epoch Loss: {:.6f} cls Loss: {:.6f} bbox Loss: {:.6f} Accuracy: {:.6f}".format(
            phase, total_loss_.avg.item(), cls_loss_.avg.item(), bbox_loss_.avg.item(), accuracy_.avg.item()))
        
        if phase == 'val' and total_loss_.avg < self.best_val_loss:
            self.best_val_loss = total_loss_.avg
            paddle.save(self.model.state_dict(), './weights/best_pnet.pdparams')
        
        return cls_loss_.avg, bbox_loss_.avg, total_loss_.avg, accuracy_.avg
    
