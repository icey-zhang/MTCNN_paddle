import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Loss:
    """Losses for classification, face box regression, landmark regression"""
    def __init__(self):
        # loss function
        # self.loss_cls = nn.BCELoss().to(device) use this loss for sigmoid score
        self.loss_cls = nn.CrossEntropyLoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()


    def cls_loss(self, gt_label, pred_label):
        # get the mask element which >= 0, only 0 and 1 can effect the detection loss
        # kind of confused here, maybe its related to cropped data state
        pred_label = paddle.squeeze(pred_label)
        mask_cls = gt_label >= 0 #zeros
        valid_gt_label = paddle.masked_select(gt_label,mask_cls)
        pred_label_t = pred_label.transpose(perm = [1,0])
        mask_cls_e = paddle.expand_as(mask_cls,pred_label_t)
        valid_pred_label = paddle.masked_select(pred_label_t,mask_cls_e).reshape([pred_label.shape[-1],-1]).transpose(perm = [1,0])
        # mask = paddle.ge(gt_label, 0) # mask is a BoolTensor, select indexes greater or equal than 0
        # valid_gt_label = paddle.masked_select(gt_label, mask)# .float()
        # #valid_pred_label = torch.masked_select(pred_label, mask)
        # valid_pred_label = pred_label[mask, :]
        return self.loss_cls(valid_pred_label, valid_gt_label)

    def box_loss(self, gt_label, gt_offset,pred_offset):
        # get the mask element which != 0
        # mask = paddle.ne(gt_label, 0)
        
        # # convert mask to dim index
        # chose_index = paddle.nonzero(mask)
        # chose_index = paddle.squeeze(chose_index)
        
        # # only valid element can effect the loss
        # valid_gt_offset = gt_offset[chose_index,:]
        # valid_pred_offset = pred_offset[chose_index,:]
        # valid_pred_offset = paddle.squeeze(valid_pred_offset)
        pred_offset=paddle.squeeze(pred_offset)
        mask_offset = gt_label != 0 #zeros
        gt_offset_t = gt_offset.transpose(perm = [1,0])
        mask_offset_e = paddle.expand_as(mask_offset,gt_offset_t)
        valid_gt_offset = paddle.masked_select(gt_offset_t,mask_offset_e).reshape([gt_offset.shape[-1],-1]).transpose(perm = [1,0])
        # valid_gt_offset = paddle.masked_select(gt_offset,mask_offset)
        pred_offset_t = pred_offset.transpose(perm = [1,0])
        mask_offset_e = paddle.expand_as(mask_offset,pred_offset_t)
        valid_pred_offset = paddle.masked_select(pred_offset_t,mask_offset_e).reshape([pred_offset.shape[-1],-1]).transpose(perm = [1,0])
        return self.loss_box(valid_pred_offset,valid_gt_offset)


    def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
        # mask = paddle.eq(gt_label,-2)
        
        # chose_index = paddle.nonzero(mask.data)
        # chose_index = paddle.squeeze(chose_index)

        # valid_gt_landmark = gt_landmark[chose_index, :]
        # valid_pred_landmark = pred_landmark[chose_index, :]
        #return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)
        # print('gt_landmark',gt_landmark)
        # print(pred_landmark)
        mask_lm = gt_label == -2 #zeros
        gt_landmark_t = gt_landmark.transpose(perm = [1,0])
        mask_lm_e = paddle.expand_as(mask_lm,gt_landmark_t)
        valid_gt_landmark = paddle.masked_select(gt_landmark_t,mask_lm_e).reshape([gt_landmark.shape[-1],-1]).transpose(perm = [1,0])
        pred_landmark_t = pred_landmark.transpose(perm = [1,0])
        mask_lm_e = paddle.expand_as(mask_lm,pred_landmark_t)
        valid_pred_landmark = paddle.masked_select(pred_landmark_t,mask_lm_e).reshape([pred_landmark.shape[-1],-1]).transpose(perm = [1,0])
        return self.loss_landmark(valid_pred_landmark, valid_gt_landmark)
        
