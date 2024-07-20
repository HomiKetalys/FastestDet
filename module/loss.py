import functools
import math
import torch
import torch.nn as nn

from torch.nn import functional as F
from ultralytics.utils.loss import TaskAlignedAssigner, BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import make_anchors


class DetectorLoss(nn.Module):
    def __init__(self, device):
        super(DetectorLoss, self).__init__()
        self.device = device

    def bbox_iou(self, box1, box2, eps=1e-7):
        # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
        box1 = box1.t()
        box2 = box2.t()

        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
        s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = pow(2, 0.5) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        gamma = angle_cost - 2
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        iou = iou - 0.5 * (distance_cost + shape_cost)

        return iou

    def build_target(self, preds, targets):
        if isinstance(preds, torch.Tensor):
            N, C, H, W = preds.shape
            preds_shape = preds.shape
        else:
            N, C, H, W = preds
            preds_shape = preds
        # batch存在标注的数据
        gt_box, gt_cls, ps_index = [], [], []
        # 每个网格的四个顶点为box中心点会归的基准点
        quadrant = torch.tensor([[0, 0], [1, 0],
                                 [0, 1], [1, 1]], device=self.device)

        if targets.shape[0] > 0:
            # 将坐标映射到特征图尺度上
            scale = torch.ones(6).to(self.device)
            scale[2:] = torch.tensor(preds_shape)[[3, 2, 3, 2]]
            gt = targets * scale

            # 扩展维度复制数据
            gt = gt.repeat(4, 1, 1)

            # 过滤越界坐标
            quadrant = quadrant.repeat(gt.size(1), 1, 1).permute(1, 0, 2)
            gij = gt[..., 2:4].long() + quadrant
            j = torch.where(gij < H, gij, 0).min(dim=-1)[0] > 0

            # 前景的位置下标
            gi, gj = gij[j].T
            batch_index = gt[..., 0].long()[j]
            ps_index.append((batch_index, gi, gj))

            # 前景的box
            gbox = gt[..., 2:][j]
            gt_box.append(gbox)

            # 前景的类别
            gt_cls.append(gt[..., 1].long()[j])

        return gt_box, gt_cls, ps_index

    def forward(self, preds, targets):
        # 初始化loss值
        ft = torch.cuda.FloatTensor if preds[0].is_cuda else torch.Tensor
        cls_loss, iou_loss, obj_loss = ft([0]), ft([0]), ft([0])

        # 定义obj和cls的损失函数
        BCEcls = nn.NLLLoss()
        # smmoth L1相比于bce效果最好
        BCEobj = nn.SmoothL1Loss(reduction='none')

        # 构建ground truth
        gt_box, gt_cls, ps_index = self.build_target(preds, targets)

        pred = preds.permute(0, 2, 3, 1)
        # 前背景分类分支
        pobj = pred[:, :, :, 0]
        # 检测框回归分支
        preg = pred[:, :, :, 1:5]
        # 目标类别分类分支
        pcls = pred[:, :, :, 5:]

        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj)
        factor = torch.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            # 计算检测框回归loss
            b, gx, gy = ps_index[0]
            ptbox = torch.ones((preg[b, gy, gx].shape)).to(self.device)
            ptbox[:, 0] = preg[b, gy, gx][:, 0].tanh() + gx
            ptbox[:, 1] = preg[b, gy, gx][:, 1].tanh() + gy
            ptbox[:, 2] = preg[b, gy, gx][:, 2].sigmoid() * W
            ptbox[:, 3] = preg[b, gy, gx][:, 3].sigmoid() * H


            # 计算检测框IOU loss
            iou = self.bbox_iou(ptbox, gt_box[0])
            # Filter
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]

            # 计算iou loss
            iou = iou[f]
            iou_loss = (1.0 - iou).mean()

            # 计算目标类别分类分支loss
            ps = torch.log(pcls[b, gy, gx])
            cls_loss = BCEcls(ps, gt_cls[0][f])

            # iou aware
            tobj[b, gy, gx] = iou.float()
            # 统计每个图片正样本的数量
            n = torch.bincount(b)
            factor[b, gy, gx] = (1. / (n[b] / (H * W))) * 0.25

        # 计算前背景分类分支loss
        obj_loss = (BCEobj(pobj, tobj) * factor).mean()

        # 计算总loss
        loss = (iou_loss * 8) + (obj_loss * 16) + cls_loss

        return iou_loss, obj_loss, cls_loss, loss


class v10DetectorLoss(DetectorLoss):
    def __init__(self, device, nc=80,reg_max=17, reg_scale=0.5,use_taa=False,img_s=(320,320)):
        super(v10DetectorLoss, self).__init__(device)
        self.reg_max = reg_max
        self.reg_scale = reg_scale
        if reg_max>1:
            self.proj = torch.arange(self.reg_max, dtype=torch.float, device=device)
        self.use_taa=use_taa
        self.fwd=self.forward_
        if self.use_taa:
            self.assigner = TaskAlignedAssigner(topk=10,num_classes=nc)
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
            self.fwd=self.forward_v8
        self.img_s=torch.tensor((img_s[1],img_s[0],img_s[1],img_s[0]),device=device)


    def forward_v8(self,preds,targets):
        # 初始化loss值
        ft = functools.partial(torch.tensor, device=preds.device)
        cls_loss, iou_loss, obj_loss, dfl_loss = ft([0.]), ft([0.]), ft([0.]), ft([0.])
        bs, ch, h, w = preds.shape
        pred = preds.permute(0, 2, 3, 1).view(bs,-1,ch)
        stride=self.img_s[1]/h

        # 检测框回归分支
        preg = pred[:, :, 1:(1 + 4 * self.reg_max)]
        # 目标类别分类分支
        pcls = pred[:, :, (1 + 4 * self.reg_max):]

        anchor_points = make_anchors([preds], [stride], 0.5)[0]
        ptbox = torch.ones((*preg.shape[:2],4), device=self.device)
        ptbox_dist=None
        if self.reg_max==1:
            ptbox[:,:,0:2] = (preg[:,:, 0:2].tanh() + anchor_points)*stride
            ptbox[:,:,2:4] =preg[:,: ,2:4].sigmoid() * self.img_s[:2]
            ptbox[:, :, 0:2] -= ptbox[:,:,2:4]*0.5
            ptbox[:, :, 2:4] += ptbox[:, :, 0:2]
        else:
            ptbox_dist=preg.view(bs,-1, 4, self.reg_max).softmax(3)
            pred_ltrb = ptbox_dist.matmul(self.proj)
            # x1 = gx - pred_dis[:, 0] / 2
            # y1 = gy - pred_dis[:, 1] / 2
            # x2 = gx + pred_dis[:, 2] / 2
            # y2 = gy + pred_dis[:, 3] / 2
            ptbox[:, :,0:2] = (anchor_points  - pred_ltrb[:,:, 0:2]*self.reg_scale)*stride
            ptbox[:,:, 2:4] = (anchor_points + pred_ltrb[:,:, 2:4]* self.reg_scale)*stride

        gtbox=targets[:,:,2:]
        gtbox[:,:,0:2]-=gtbox[:,:,2:4]*0.5
        gtbox[:,:,2:4]+=gtbox[:,:,0:2]
        gtbox=torch.clip(gtbox,0,1)
        gtbox=gtbox*self.img_s

        _, target_bboxes, target_scores, fg_mask, _=self.assigner.forward(
            pd_scores=pcls.sigmoid(),
            pd_bboxes=ptbox,
            anc_points=anchor_points*stride,
            gt_labels=targets[:,:,1:2],
            gt_bboxes=gtbox,
            mask_gt=targets[:,:,0:1],
        )
        target_scores_sum = max(target_scores.sum(), 1)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(ptbox[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        iou_loss = ((1.0 - iou) * weight).sum() / target_scores_sum
        cls_loss =self.bce(pcls, target_scores).sum() / target_scores_sum  # BCE

        if self.reg_max>1:
            gtbox=target_bboxes/stride
            x1y1, x2y2 = gtbox.chunk(2, -1)
            gtbox_ltrb=torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, (self.reg_max - 1) * self.reg_scale - 0.001) / self.reg_scale  # dist (lt, rb)
            dfl_loss = self._df_loss(ptbox_dist[fg_mask], gtbox_ltrb[fg_mask]) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum

        iou_loss *=7.5
        cls_loss *= 0.5
        dfl_loss*=1.5
        loss=iou_loss+cls_loss+dfl_loss
        return iou_loss, obj_loss, cls_loss, dfl_loss, loss


    def forward(self, preds, targets):

        if isinstance(preds, list):
            iou, obj, cls, dfl, total = [], [], [], [], []
            for p in preds:
                iou_, obj_, cls_, dfl_, total_ = self.fwd(p, targets)
                iou.append(iou_)
                obj.append(obj_)
                cls.append(cls_)
                dfl.append(dfl_)
                total.append(total_)
            return sum(iou) / len(preds), sum(obj) / len(preds), sum(cls) / len(preds), sum(dfl) / len(preds), sum(
                total) / len(preds)
        else:
            return self.fwd(preds, targets)


    def forward_(self, preds, targets):
        # 初始化loss值
        ft = functools.partial(torch.tensor, device=preds.device)
        cls_loss, iou_loss, obj_loss, dfl_loss = ft([0]), ft([0]), ft([0]), ft([0])

        # 定义obj和cls的损失函数
        BCEcls = nn.NLLLoss()
        # smmoth L1相比于bce效果最好
        BCEobj = nn.SmoothL1Loss(reduction='none')

        bs, ch, h, w = preds.shape
        # 构建ground truth
        gt_box, gt_cls, ps_index = self.build_target((bs, ch - 4 * self.reg_max + 4, h, w), targets)

        pred = preds.permute(0, 2, 3, 1)

        # 前背景分类分支
        pobj = pred[:, :, :, 0]
        # 检测框回归分支
        preg = pred[:, :, :, 1:(1 + 4 * self.reg_max)]
        # 目标类别分类分支
        pcls = pred[:, :, :, (1 + 4 * self.reg_max):].softmax(3)

        N, H, W, C = pred.shape
        tobj = torch.zeros_like(pobj)
        factor = torch.ones_like(pobj) * 0.75

        if len(gt_box) > 0:
            # 计算检测框回归loss
            b, gx, gy = ps_index[0]
            ptbox = torch.ones(preg[b, gy, gx].shape, device=self.device)
            pred_distri = preg[b, gy, gx]
            if self.reg_max == 1:
                ptbox[:, 0] = pred_distri[:, 0].tanh() + gx
                ptbox[:, 1] = pred_distri[:, 1].tanh() + gy
                ptbox[:, 2] = pred_distri[:, 2].sigmoid() * W
                ptbox[:, 3] = pred_distri[:, 3].sigmoid() * H
            else:
                pred_ltrb = pred_distri.view(-1, 4, self.reg_max).softmax(2).matmul(self.proj)
                ptbox[:, 0] = gx + (pred_ltrb[:, 2] - pred_ltrb[:, 0]) * self.reg_scale * 0.5
                ptbox[:, 1] = gy + (pred_ltrb[:, 3] - pred_ltrb[:, 1]) * self.reg_scale * 0.5
                ptbox[:, 2] = (pred_ltrb[:, 2] + pred_ltrb[:, 0]) * self.reg_scale
                ptbox[:, 3] = (pred_ltrb[:, 3] + pred_ltrb[:, 1]) * self.reg_scale

            # 计算检测框IOU loss
            iou = self.bbox_iou(ptbox, gt_box[0])
            # Filter
            f = iou > iou.mean()
            b, gy, gx = b[f], gy[f], gx[f]
            iou = iou[f]
            pred_distri = pred_distri[f]
            gt_box[0] = gt_box[0][f]
            gt_cls[0]=gt_cls[0][f]

            if self.reg_max>1:
                gt_box[0] = torch.cat((gx[:, None] - gt_box[0][:, 0:1] + gt_box[0][:, 2:3] / 2,
                                  gy[:, None] - gt_box[0][:, 1:2] + gt_box[0][:, 3:4] / 2,
                                  gt_box[0][:, 0:1] + gt_box[0][:, 2:3] / 2 - gx[:, None],
                                  gt_box[0][:, 1:2] + gt_box[0][:, 3:4] / 2 - gy[:, None],
                                  ), -1).clamp_(0, (
                        self.reg_max - 1) * self.reg_scale - 0.001) / self.reg_scale  # dist (lt, rb)

                dfl_loss = self._df_loss(pred_distri, gt_box[0])
                dfl_loss = dfl_loss.mean()

            # 计算iou loss
            iou_loss = (1.0 - iou).mean()

            # 计算目标类别分类分支loss
            ps = torch.log(pcls[b, gy, gx])
            cls_loss = BCEcls(ps, gt_cls[0])

            # iou aware
            tobj[b, gy, gx] = iou.float()
            # 统计每个图片正样本的数量
            n = torch.bincount(b)
            factor[b, gy, gx] = (1. / (n[b] / (H * W))) * 0.25

        # 计算前背景分类分支loss
        obj_loss = (BCEobj(pobj, tobj) * factor).mean()

        # 计算总loss
        loss = (iou_loss * 8) + (obj_loss * 16) + cls_loss + (dfl_loss * 1.6)

        return iou_loss, obj_loss, cls_loss, dfl_loss, loss

    def _df_loss(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist.view(-1,self.reg_max), tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist.view(-1,self.reg_max), tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)
