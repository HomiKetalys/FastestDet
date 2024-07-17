import torch
import numpy as np
from tqdm import tqdm
from utils.tool import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoDetectionEvaluator():
    def __init__(self, names, device,cfg,norm=True):
        self.device = device
        self.classes = []
        self.cfg=cfg
        with open(names, 'r') as f:
            for line in f.readlines():
                self.classes.append(line.strip())
        self.norm=norm
    
    def coco_evaluate(self, gts, preds):
        # Create Ground Truth
        coco_gt = COCO()
        coco_gt.dataset = {}
        coco_gt.dataset["images"] = []
        coco_gt.dataset["annotations"] = []
        k = 0
        for i, gt in enumerate(gts):
            for j in range(gt.shape[0]):
                k += 1
                coco_gt.dataset["images"].append({"id": i})
                coco_gt.dataset["annotations"].append({"image_id": i, "category_id": gt[j, 0],
                                                    "bbox": np.hstack([gt[j, 1:3], gt[j, 3:5] - gt[j, 1:3]]),
                                                    "area": np.prod(gt[j, 3:5] - gt[j, 1:3]),
                                                    "id": k, "iscrowd": 0})
                
        coco_gt.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_gt.createIndex()

        # Create preadict 
        coco_pred = COCO()
        coco_pred.dataset = {}
        coco_pred.dataset["images"] = []
        coco_pred.dataset["annotations"] = []
        k = 0
        for i, pred in enumerate(preds):
            for j in range(pred.shape[0]):
                k += 1
                coco_pred.dataset["images"].append({"id": i})
                coco_pred.dataset["annotations"].append({"image_id": i, "category_id": int(pred[j, 0]),
                                                        "score": pred[j, 1], "bbox": np.hstack([pred[j, 2:4], pred[j, 4:6] - pred[j, 2:4]]),
                                                        "area": np.prod(pred[j, 4:6] - pred[j, 2:4]),
                                                        "id": k})
                
        coco_pred.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        coco_pred.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.params.iouThrs = np.linspace(self.cfg.iou, 0.95, int(np.round((0.95 - .4) / .05)) + 1, endpoint=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP05 = coco_eval.stats[1]

        # 过gt和pred计算每个类别的recall
        precisions = coco_eval.eval['precision']  # TP/(TP+FP) right/detection
        recalls = coco_eval.eval['recall']  # iou*class_num*Areas*Max_det TP/(TP+FN) right/gt
        print(
            '\nIOU:{} MAP:{:.4f} Recall:{:.4f}'.format(coco_eval.params.iouThrs[0], np.mean(precisions[0, :, :, 0, -1]),
                                                       np.mean(recalls[0, :, 0, -1])))
        # Compute per-category AP
        # from https://github.com/facebookresearch/detectron2/
        # precision: (iou, recall, cls, area range, max dets)
        results_per_category = []
        results_per_category_ = []
        for idx, catId in enumerate(range(len(self.classes))):
            name = self.classes[idx]
            precision = precisions[:, :, idx, 0, -1]
            precision_ = precisions[0, :, idx, 0, -1]
            precision = precision[precision > -1]

            recall = recalls[:, idx, 0, -1]
            recall_ = recalls[0, idx, 0, -1]
            recall = recall[recall > -1]

            if precision.size:
                ap = np.mean(precision)
                ap_ = np.mean(precision_)
                rec = np.mean(recall)
                rec_ = np.mean(recall_)
            else:
                ap = float('nan')
                ap_ = float('nan')
                rec = float('nan')
                rec_ = float('nan')
            res_item = [f'{name}', f'{float(ap):0.4f}', f'{float(rec):0.4f}']
            results_per_category.append(res_item)
            res_item_ = [f'{name}', f'{float(ap_):0.4f}', f'{float(rec_):0.4f}']
            results_per_category_.append(res_item_)
        print(results_per_category_)
        return mAP05

    def compute_map(self, val_dataloader, model,cfg):
        gts, pts = [], []
        pbar = tqdm(val_dataloader)
        for i, (imgs, targets) in enumerate(pbar):
            # 数据预处理
            imgs = imgs.to(self.device).float() / 255.0
            with torch.no_grad():
                # 模型预测
                preds = model(imgs)
                # 特征图后处理
                output = handle_preds(preds, self.device,0.001, cfg=cfg,norm=self.norm)

            # 检测结果
            N, _, H, W = imgs.shape
            for p in output:
                pbboxes = []
                for b in p:
                    b = b.cpu().numpy()
                    score = b[4]
                    category = b[5]
                    x1, y1, x2, y2 = b[:4] * [W, H, W, H]
                    pbboxes.append([category, score, x1, y1, x2, y2])
                pts.append(np.array(pbboxes))

            # 标注结果
            for n in range(N):
                tbboxes = []
                for t in targets:
                    if t[0] == n:
                        t = t.cpu().numpy()
                        category = t[1]
                        bcx, bcy, bw, bh = t[2:] * [W, H, W, H]
                        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
                        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
                        tbboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(tbboxes))
                
        mAP05 = self.coco_evaluate(gts, pts)

        return mAP05
