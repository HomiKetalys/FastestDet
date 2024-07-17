import os.path

import yaml
import torch
import torchvision


# 解析yaml配置文件
class LoadYaml:
    def __init__(self, path):
        with open(path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]

        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]

        self.separation = data["MODEL"]["SEPARATION"]
        self.separation_scale = data["MODEL"]["SEPARATION_SCALE"]

        self.reg_max = data["MODEL"]["REG_MAX"]
        self.reg_scale = data["MODEL"]["REG_SCALE"]

        self.category_num = data["MODEL"]["NC"]

        self.amp=data["TRAIN"]["AMP"]

        self.conf=data["VAL"]["CONF"]
        self.nms=data["VAL"]["NMS"]
        self.iou=data["VAL"]["IOU"]
        self.label_flag=os.path.splitext(os.path.split(path)[1])[0]

        print("Load yaml sucess...")


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
#     total_bboxes, output_bboxes = [], []
#     reg_max = 17
#     # 将特征图转换为检测框的坐标
#     N, C, H, W = preds.shape
#     bboxes = torch.zeros((N, H, W, 6))
#     pred = preds.permute(0, 2, 3, 1)
#     # 前背景分类分支
#     pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
#     # 检测框回归分支
#     preg = pred[:, :, :, 1:(1 + 4 * reg_max)]
#     # 目标类别分类分支
#     pcls = pred[:, :, :, (1 + 4 * reg_max):]
#
#     # 检测框置信度
#     bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
#     bboxes[..., 5] = pcls.argmax(dim=-1)
#
#     # 检测框的坐标
#     gy, gx = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing="ij")
#     # bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
#     # bcx = (preg[..., 0].tanh() + gx) / W
#     # bcy = (preg[..., 1].tanh() + gy) / H
#     #
#     # # cx,cy,w,h = > x1,y1,x2,y1
#     # x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
#     # x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
#
#     pred_distri = preg.view(N, H, W, 4, reg_max)
#     proj = torch.arange(reg_max, dtype=torch.float, device=device)
#     pred_ltrb = pred_distri.softmax(4).matmul(proj.type(preg.dtype))
#     x1 = (gx + 0.5 - pred_ltrb[..., 0] / 2) / W
#     y1 = (gy + 0.5 - pred_ltrb[..., 1] / 2) / H
#     x2 = (gx + 0.5 + pred_ltrb[..., 2] / 2) / W
#     y2 = (gy + 0.5 + pred_ltrb[..., 3] / 2) / H
#
#     bboxes[..., 0], bboxes[..., 1] = x1, y1
#     bboxes[..., 2], bboxes[..., 3] = x2, y2
#     bboxes = bboxes.reshape(N, H * W, 6)
#     total_bboxes.append(bboxes)
#
#     batch_bboxes = torch.cat(total_bboxes, 1)
#
#     # 对检测框进行NMS处理
#     for p in batch_bboxes:
#         output, temp = [], []
#         b, s, c = [], [], []
#         # 阈值筛选
#         t = p[:, 4] > conf_thresh
#         pb = p[t]
#         for bbox in pb:
#             obj_score = bbox[4]
#             category = bbox[5]
#             x1, y1 = bbox[0], bbox[1]
#             x2, y2 = bbox[2], bbox[3]
#             s.append([obj_score])
#             c.append([category])
#             b.append([x1, y1, x2, y2])
#             temp.append([x1, y1, x2, y2, obj_score, category])
#         # Torchvision NMS
#         if len(b) > 0:
#             b = torch.Tensor(b).to(device)
#             c = torch.Tensor(c).squeeze(1).to(device)
#             s = torch.Tensor(s).squeeze(1).to(device)
#             keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
#             for i in keep:
#                 output.append(temp[i])
#         output_bboxes.append(torch.Tensor(output))
#     return output_bboxes


# 后处理(归一化后的坐标)
def handle_preds_(preds, device, cfg=None,norm=True):
    total_bboxes  = []
    # 将特征图转换为检测框的坐标
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)
    if cfg is None:
        reg_max = 1
        reg_scale = 1
    else:
        reg_max = cfg.reg_max
        reg_scale = cfg.reg_scale
    # 前背景分类分支
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    # 检测框回归分支
    preg = pred[:, :, :, 1:(1 + 4 * reg_max)]
    # 目标类别分类分支
    pcls = pred[:, :, :, (1 + 4 * reg_max):]

    # 检测框置信度
    if pcls.shape[3]>0:
        bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
        bboxes[..., 5] = pcls.argmax(dim=-1)
    else:
        bboxes[..., 4] = pobj.squeeze(-1)**0.6
        bboxes[..., 5] = 0


    # 检测框的坐标
    gy, gx = torch.meshgrid([torch.arange(H, device=device), torch.arange(W, device=device)], indexing="ij")
    if reg_max == 1:
        if norm:
            bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
            bcx = (preg[..., 0].tanh() + gx) / W
            bcy = (preg[..., 1].tanh() + gy) / H
        else:
            bw, bh = preg[..., 2], preg[..., 3]
            bcx = (preg[..., 0] + gx) / W
            bcy = (preg[..., 1] + gy) / H

        # cx,cy,w,h = > x1,y1,x2,y1
        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
    else:
        pred_distri = preg.view(N, H, W, 4, reg_max)
        proj = torch.arange(reg_max, dtype=torch.float, device=device)
        if norm:
            pred_distri = pred_distri.softmax(4)
        pred_ltrb=pred_distri.matmul(proj.type(preg.dtype))
        x1 = (gx - pred_ltrb[..., 0] * reg_scale) / W
        y1 = (gy - pred_ltrb[..., 1] * reg_scale) / H
        x2 = (gx + pred_ltrb[..., 2] * reg_scale) / W
        y2 = (gy + pred_ltrb[..., 3] * reg_scale) / H

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H * W, 6)
    total_bboxes.append(bboxes)

    batch_bboxes = torch.cat(total_bboxes, 1)
    return batch_bboxes

def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.5, cfg=None,norm=True):
    output_bboxes=[]
    if isinstance(preds,list):
        batch_bboxes = []
        for p in preds:
            batch_bboxes.append(handle_preds_(p,device,cfg,norm))
        batch_bboxes=torch.cat(batch_bboxes,1)
    else:
        batch_bboxes=handle_preds_(preds,device,cfg,norm)
    if cfg is not None:
        nms_thresh=cfg.nms
    # 对检测框进行NMS处理
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # 阈值筛选
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes
