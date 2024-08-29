import math
import os
import cv2
import numpy as np

import torch
import random


def contrast_and_brightness(img, boxes, prob=0.1):
    if random.randint(0, 99) < int(prob * 100):
        alpha = random.uniform(0.25, 1.75)
        beta = random.uniform(0.25, 1.75)
        blank = np.zeros(img.shape, img.dtype)
        # dst = alpha * img + beta * blank
        dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
        return dst, boxes
    else:
        return img, boxes


def motion_blur(image, boxes, prob=0.1):
    if random.randint(0, 99) < int(prob * 100):
        degree = random.randint(3, 25)
        angle = random.uniform(-360, 360)
        image = np.array(image)

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred, boxes
    else:
        return image, boxes


def augment_hsv(img, boxes, hgain=0.0138, sgain=0.678, vgain=0.36, prob=0.1):
    if random.randint(0, 99) < int(prob * 100):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        return img, boxes
    else:
        return img, boxes


def random_crop(image, boxes, prob=0.1):
    if random.randint(0, 99) < int(prob * 100):
        height, width, _ = image.shape
        # random crop imgage
        cw, ch = random.randint(int(width * 0.75), width), random.randint(int(height * 0.75), height)
        cx, cy = random.randint(0, width - cw), random.randint(0, height - ch)

        roi = image[cy:cy + ch, cx:cx + cw]
        roi_h, roi_w, _ = roi.shape

        output = []
        for box in boxes:
            index, category = box[0], box[1]
            bx, by = box[2] * width, box[3] * height
            bw, bh = box[4] * width, box[5] * height

            bx, by = (bx - cx) / roi_w, (by - cy) / roi_h
            bw, bh = bw / roi_w, bh / roi_h

            output.append([index, category, bx, by, bw, bh])

        output = np.array(output, dtype=float)

        return roi, output
    else:
        return image, boxes


def random_narrow(image, boxes, prob=0.1):
    if random.randint(0, 99) < int(prob * 100):
        height, width, _ = image.shape
        # random narrow
        cw, ch = random.randint(width, int(width * 1.25)), random.randint(height, int(height * 1.25))
        cx, cy = random.randint(0, cw - width), random.randint(0, ch - height)

        background = np.ones((ch, cw, 3), np.uint8) * 128
        background[cy:cy + height, cx:cx + width] = image

        output = []
        for box in boxes:
            index, category = box[0], box[1]
            bx, by = box[2] * width, box[3] * height
            bw, bh = box[4] * width, box[5] * height

            bx, by = (bx + cx) / cw, (by + cy) / ch
            bw, bh = bw / cw, bh / ch

            output.append([index, category, bx, by, bw, bh])

        output = np.array(output, dtype=float)

        return background, output
    else:
        return image, boxes


def random_flip_lr(image, boxes, prob=0.5):
    if random.randint(0, 99) < int(prob * 100):
        image = image[:, ::-1, :]

        output = []
        for box in boxes:
            index, category = box[0], box[1]
            bx, by = box[2], box[3]
            bw, bh = box[4], box[5]
            bx, by = 1 - bx, 1 - by
            output.append([index, category, bx, by, bw, bh])
        output = np.array(output, dtype=float)
        return image, output
    else:
        return image, boxes


def img_aug(img, boxes):
    img, boxes = contrast_and_brightness(img, boxes, prob=0.4)
    img, boxes = augment_hsv(img, boxes, prob=0.4)
    img, boxes = motion_blur(img, boxes, prob=0.2)
    img, boxes = random_narrow(img, boxes, prob=0.2)
    img, boxes = random_crop(img, boxes, prob=0.2)
    img, boxes = random_flip_lr(img, boxes, prob=0.5)
    return img, boxes


def collate_fnt(batch):
    img, label = zip(*batch)
    max_len = max([len(lb) for lb in label])
    gts = []
    for i, l in enumerate(label):
        gt = torch.zeros((max_len, 6))
        if len(l) > 0:
            gt[:len(l), :] = l
        gts.append(gt)
    return torch.stack(img), torch.stack(gts, 0)


def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)


class TensorDataset():
    def __init__(self, path, img_width, img_height, aug=False, label_flag="coco_80"):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.aug = aug
        self.path = path
        self.data_list = []
        self.img_width = img_width
        self.img_height = img_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
        self.cache = False

        # 数据检查
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if not os.path.isabs(data_path):
                    data_path = os.path.join(os.path.split(self.path)[0], data_path)
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        label_path = os.path.splitext(data_path)[0] + ".txt"
                        if os.path.exists(label_path):
                            # img = cv2.imread(data_path)
                            self.data_list.append((data_path, label_path))
                        else:

                            # if not self.aug:
                            #     label_flag="coco_80"
                            label_path = os.path.relpath(label_path, os.path.split(self.path)[0])
                            label_path = os.path.join(os.path.split(self.path)[0], label_flag,
                                                      os.path.join(*label_path.split(os.path.sep)[1:]))
                            # label_path=label_path.replace("images",label_flag)
                            if os.path.exists(label_path):
                                # img = cv2.imread(data_path)
                                self.data_list.append((data_path, label_path))
                            else:
                                raise Exception("%s is not exist" % label_path)
                else:
                    raise Exception("%s is not exist" % data_path)

    def __getitem__(self, index):
        img_path, label_path = self.data_list[index]
        img_path_c = img_path.replace("images", "cache")

        # 加载label文件
        if os.path.exists(label_path):
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([1, l[0], l[1], l[2], l[3], l[4]])
            label = np.array(label, dtype=np.float32)

            if label.shape[0]:
                assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                # assert (label >= 0).all(), 'negative labels: %s'%label_path
                # assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path)

        # 加载图片
        if self.cache:
            if os.path.exists(img_path_c):
                img = cv2.imread(img_path_c)
            else:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(img_path_c, img)
        else:
            img = cv2.imread(img_path)
            # 是否进行数据增强
            if self.aug:
                # img = img[0::2, 0::2, :]
                if random.randint(1, 10) % 2 == 0:
                    img, label = random_narrow(img, label,1)
                else:
                    img, label = random_crop(img, label,1)
                # img, label = img_aug(img, label)
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # debug
        # for box in label:
        #     bx, by, bw, bh = box[2], box[3], box[4], box[5]
        #     x1, y1 = int((bx - 0.5 * bw) * self.img_width), int((by - 0.5 * bh) * self.img_height)
        #     x2, y2 = int((bx + 0.5 * bw) * self.img_width), int((by + 0.5 * bh) * self.img_height)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.imwrite("debug.jpg", img)

        img = img.transpose(2, 0, 1)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    data = TensorDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)
