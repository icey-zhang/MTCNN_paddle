import paddle

def compute_accuracy(prob_cls, gt_cls):
    # we only need the detection which >= 0
    prob_cls = paddle.squeeze(prob_cls)
    mask = gt_cls>= 0
    
    # get valid elements
    valid_gt_cls = paddle.masked_select(gt_cls,mask)
    pred_label_t = prob_cls.transpose(perm = [1,0])
    mask_e = paddle.expand_as(mask,pred_label_t)
    valid_prob_cls = paddle.masked_select(pred_label_t,mask_e).reshape([prob_cls.shape[-1],-1]).transpose(perm = [1,0])
    # valid_gt_cls = gt_cls[mask]
    # valid_prob_cls = prob_cls[mask]
    size = min(valid_gt_cls.shape[0], valid_prob_cls.shape[0])
    
    # get max index with softmax layer
    valid_pred_cls = paddle.argmax(valid_prob_cls, axis=1)
    
    # right_ones = paddle.eq(valid_pred_cls, valid_gt_cls)
    right_ones = paddle.to_tensor(valid_pred_cls == valid_gt_cls,dtype='float32').sum()
    
    return paddle.divide(right_ones,(paddle.to_tensor(size,dtype='float32') + 1e-16))