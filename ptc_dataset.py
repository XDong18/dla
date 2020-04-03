import torch
from PIL import Image
import os
import cv2
from os import listdir
import logging
import numpy as np 
from torch.utils.data import Dataset
from glob import glob


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = []
        files = listdir(imgs_dir)
        files.sort()
        for file in files:
            sub_img_dir = os.path.join(imgs_dir, file)
            image_files = listdir(sub_img_dir)
            image_files.sort()
            sub_ids = [os.path.join(file, image_f.split('.')[0]) for image_f in image_files]
            self.ids = self.ids + sub_ids
    
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_image(cls, img):
        img = cv2.resize(img, (1920, 1216))
        img = 255 - img
        img[np.where(img == 255)] = 0
        dila_point = np.logical_or.reduce(img > 0, axis=2)
        point_list = np.argwhere(dila_point==True)
        for point in point_list:
            cv2.circle(img, (point[1], point[0]), 3, 
            (int(img[point[0], point[1], 0]), int(img[point[0], point[1], 1]), int(img[point[0], point[1], 2])), -1)
        
        img_new = np.array([img[:, :, 0]])

        if img_new.max() > 1:
            img_new = img_new / 255

        return img_new
    
    @classmethod
    def preprocess_mask(cls, img, ignore_mask):
        img = cv2.resize(img, (1920, 1216))
        dila_point = np.logical_or.reduce(img > 0, axis=2)
        point_list = np.argwhere(dila_point==True)
        for point in point_list:
            cv2.circle(img, (point[1], point[0]), 3, 
            (int(img[point[0], point[1], 0]), int(img[point[0], point[1], 1]), int(img[point[0], point[1], 2])), -1)

        valid_point = np.logical_or.reduce(img > 0, axis=2)
        new_img = np.zeros((img.shape[0], img.shape[1]),dtype=np.float32)
        new_img[np.where(valid_point)] = 1
        new_img[np.where(ignore_mask)] = -1

        return new_img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = cv2.imread(mask_file[0])
        # img = cv2.imread(img_file[0])
        mask = np.load(mask_file[0])
        # img = np.load(img_file[0]) #TODO
        img = Image.open(img_file[0])
        img = img.resize((1920, 1216))
        img = np.array(img)
        img = np.transpose(img, (2,0,1))
        if img.max()>1:
            img = img / 255.
        # img = np.array([img])
        # print(img_file[0])


        # img = cv2.resize(img, (1216, 1920))
        # mask = cv2.resize(mask, (1216, 1920))

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # img = self.preprocess_image(img)
        # ignore_mask = img[0]==0
        # mask = self.preprocess_mask(mask, ignore_mask)

        # _, h, w = img.shape
        # img = img[:, int(h / 4):int(3 * h / 4), 0:int(3 * w/8)]
        # mask = mask[int(h / 4):int(3 * h / 4), 0:int(3 * w/8)]

        return(tuple([torch.from_numpy(img), torch.from_numpy(mask)]))
        # return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'ignore_mask': torch.from_numpy(ignore_mask)}

if __name__ =='__main__':
    val_dir_img = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/npy_img/'
    val_dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/npy_mask/'
    my_val = BasicDataset(val_dir_img, val_dir_mask)
    my_val.__getitem__(709)