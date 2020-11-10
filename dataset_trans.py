import torch
from PIL import Image
import os
import cv2
from os import listdir
import logging
import numpy as np 
from torch.utils.data import Dataset
from glob import glob
import data_transforms as transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, transforms, is_train=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.ids = []
        self.transforms = transforms
        self.is_train = is_train
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
        if self.is_train:
            mask = np.load(mask_file[0])
             # TODO
            mask[mask==2] = 0
            mask[mask==3] = 0
            mask[mask==5] = 0
            mask[mask==6] = 0
            mask[mask==8] = 0
            mask[mask==10] = 0
            mask[mask==11] = 0
            mask[mask==12] = 0
            mask[mask==13] = 0
            mask[mask==15] = 0
            mask[mask==16] = 0

            mask[mask==4] = 2
            mask[mask==7] = 3
            mask[mask==9] = 4
            mask[mask==14] = 1
            mask[mask==17] = 3
            ##################################################
            mask = mask + 100
            # mask = np.expand_dims(mask, 2)
            # mask = np.concatenate((mask, mask, mask), axis=2)
            mask_PIL = Image.fromarray(mask)
            # img = np.load(img_file[0]) #TODO
            img = Image.open(img_file[0])
            # img = img.resize((1920, 1216)) # TODO resize is not necessary
            data = [img]
            data.append(mask_PIL)
            data = list(self.transforms(*data))
            trans_temp = transforms.ToTensor()
            tensor_img = trans_temp(data[0])[0]
            array_mask = np.array(data[1]) - 100
            # img = np.array(img)
            # img = np.transpose(img, (2,0,1))
            # if img.max()>1:
            #     img = img / 255.
            # data[1] = data[1][0]
            return(tuple([tensor_img, torch.from_numpy(array_mask)]))
        else:
            mask = np.load(mask_file[0])
            
            # TODO
            mask[mask==2] = 0
            mask[mask==3] = 0
            mask[mask==5] = 0
            mask[mask==6] = 0
            mask[mask==8] = 0
            mask[mask==10] = 0
            mask[mask==11] = 0
            mask[mask==12] = 0
            mask[mask==13] = 0
            mask[mask==15] = 0
            mask[mask==16] = 0

            mask[mask==4] = 2
            mask[mask==7] = 3
            mask[mask==9] = 4
            mask[mask==14] = 1
            mask[mask==17] = 3

            img = Image.open(img_file[0])
            # img = img.resize((1920, 1216)) # TODO resize is not necessary
            img = np.array(img)
            img = np.transpose(img, (2,0,1))
            if img.max()>1:
                img = img / 255.
            return(tuple([torch.from_numpy(img), torch.from_numpy(mask)]))

        

if __name__ =='__main__':
    val_dir_img = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/npy_img/'
    val_dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/npy_mask/'
    my_val = BasicDataset(val_dir_img, val_dir_mask)
    my_val.__getitem__(709)