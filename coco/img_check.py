from pycocotools.coco import COCO
import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
import data_transforms as transforms



COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',*
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = { 1:  1,  2:  2,  3:  3,  4:  4,  5:  5,  6:  6,  7:  7,  8:  8,
                   9:  9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


class COCOSeg(data.Dataset):
    def __init__(self, image_path, info_file, transforms=None, is_train=False):
        self.root = image_path
        self.coco = COCO(info_file)
        self.ids = list(self.coco.imgToAnns.keys()) # may be some imaegs don't have labels
        self.transforms = transforms
        self.is_train is_train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img, mask = self.pull_item(index)
        if self.is_train:
            mask_PIL = Image.fromarray(mask)
            # img = img.resize((1920, 1216))
            data = [img]
            data.append(mask_PIL)
            data = list(self.transforms(*data))
            trans_temp = transforms.ToTensor()
            tensor_img = trans_temp(data[0])[0]
            array_mask = np.array(data[1])
            return(tuple([tensor_img, torch.from_numpy(array_mask)]))
        else:
            # img = img.resize((1920, 1216))
            img = np.array(img)
            img = np.transpose(img, (2,0,1))
            if img.max()>1:
                img = img / 255.
            return(tuple([torch.from_numpy(img), torch.from_numpy(mask)]))

    def pull_item(self, index):
        '''
        Return Image, array
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target = self.coco.loadAnns(ann_ids)

        crowd = [x for x in target if ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]

        target = crowd + target

        file_name = self.coco.loadImgs(img_id)[0]['file_name']
        path = osp.join(self.root, file_name)
        img = Image.open(path)

        width, height = img.size
        mask = np.zerso((height, width))
        for obj in target:
            cat_id = COCO_LABEL_MAP[obj['category_id']] - 1
            obj_mask = self.coco.annToMask(obj)
            mask[np.where(obj_mask==1)] = cat_id
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)
        return img, mask
