# -*- coding: utf-8 -*-
# 源码地址 https://github.com/amdegroot/ssd.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing 位置预测, 置信度预测,
            and 先验anchor from SSD net.
                置信度预测conf shape: torch.size(batch_size,num_priors 8732,num_classes)
                位置预测loc shape: torch.size(batch_size,num_priors 8732,4)
                先验anchor priors shape: torch.size(num_priors,4)
            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (维度5：四个坐标+置信度).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)    # batch_size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))    # 8732
        num_classes = self.num_classes    # 21

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data    # 前四个为坐标
            labels = targets[idx][:, -1].data    # 最后一个为置信度
            defaults = priors.data
            # 将GT框与先验anchor的坐标与label进行match，即为每一个先验anchor指定一个label，并并计算与匹配到的GT框的坐标偏差
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets Variable对Tensor对象进行封装，requires_grad控制着是否在反向传播过程中对该节点求梯度
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0    # 排除label为0的背景，shape[batch_size, 8732], true or false
        num_pos = pos.sum(dim=1, keepdim=True)    # 包含的目标总数

        # Localization Loss (Smooth L1) 定位损失函数
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)    # pos扩充一个维度[batch_size,8732]——>[batch_size,8732,1]，在expand_as[batch_size,8732,4]，4个维度数据相同
        loc_p = loc_data[pos_idx].view(-1, 4)    # 取出带目标的框，shape[num_pos,4]
        loc_t = loc_t[pos_idx].view(-1, 4)    # shape[num_pos,4]
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)    # Huber损失函数

        # 为难样本挖掘计算 max conf across batch（batch中最大置信度？）
        batch_conf = conf_data.view(-1, self.num_classes)    # shape[batch_size*8732,21]
        # batch_conf.gather(1, conf_t.view(-1, 1))得到每个先验anchor在匹配GT框后的label，shape[8732, 1]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # 难样本挖掘
        # 本质上就是将所有负例的loss按照降序排列后选取前n个
        loss_c[pos] = 0  # 暂时过滤掉正样本
        loss_c = loss_c.view(num, -1)    # shape[batch_size,8732]
        _, loss_idx = loss_c.sort(1, descending=True)    # 降序排序，得到index
        _, idx_rank = loss_idx.sort(1)    # 升序
        # 这样两次排序以后，所得的idx_rank为原loss_c中数值大小的整数映射，即原数大，映射的整数就大，反之则小
        
        num_pos = pos.long().sum(1, keepdim=True)    # 包含目标的数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)    # 按照预定比例随机生成负例数量（正3:负1）
        neg = idx_rank < num_neg.expand_as(idx_rank)    # 选择负例，shape[batch_size,8732]，true or false

        # 置信度损失 Including Positive and Negative Examples
        # shape[batch_size, 8732, 21] true or false, true代表选择出来用于训练的样例，其index为label
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        # torch.gt(0)逐个元素与0比较,大则true,否则false
        # conf_data[(pos_idx + neg_idx).gt(0)]取出所有用于训练的样例
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]    # 真实值
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
        

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    # step 1：计算IOU矩阵 shape:[num_object,num_priors]
    overlaps = jaccard(
        truths,
        point_form(priors)    # 返回框的xmin ymin xmax ymax
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    # step 2：为每个GT框匹配一个IOU最大的先验anchor
    # best_prior_overlap 保存每个GT框匹配到的最大IOU shape[num_objects,1]
    # best_prior_idx 保存匹配到的先验anchor的索引 取值范围[0,num_priors]
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    
    # step 3：为每个先验anchor匹配一个IOU最大的GT框
    # best_truth_overlap 保存每个先验anchor匹配到最大IOU的GT框 shape[1,num_priors]
    # best_truth_idx 保存匹配到的GT框的索引 取值范围[0,num_objects]
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    
    best_truth_idx.squeeze_(0)    # [num_objects]
    best_truth_overlap.squeeze_(0)    # [num_objects]
    best_prior_idx.squeeze_(1)    # [8732]
    best_prior_overlap.squeeze_(1)    # [8732]
    
    # step 4：确定每个先验anchor的最佳匹配GT框
    # index_fill_ 将best_truth_overlap中所有best_prior_idx正例对应的索引位置的值置为2
    # IOU取值0~1，置为2即肯定被选中（以防再step6中被置为背景类？）
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    # 确保每个GT框都有一个最佳匹配先验anchor
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
        
    # step 5：取出与每个anchor最佳匹配的GT框
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    
    # step 6：类别，因为背景设为0，其他类别数要+1
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    # IOU小于阈值的label置为背景
    conf[best_truth_overlap < threshold] = 0  # label as background
    # step 7：位置编码（以偏差表示）
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    
    
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) 每个先验anchor对应最佳的GT框
            Shape: [num_priors, 4].
        priors: (tensor) 先验anchor，4维度分别为[xcenter, ycenter, w, h]
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """
    # 计算GT框和先验anchor的偏差用于回归
    
    # dist b/t match center and prior's center中心偏差
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    # match wh / prior wh宽高比例
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
