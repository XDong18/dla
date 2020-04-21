import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from torchvision import transforms

from ptc_dataset import BasicDataset
import dla_up
from torch.autograd import Variable
from os.path import exists, join, split, dirname
from os import listdir
import scipy.ndimage
from matplotlib import pyplot as plt
import copy


def blend_mask(mask, img):
    new_img = cv2.resize(img, (1920, 1216)).astype(np.float64)
    s_mask = copy.deepcopy(mask[:,:,0])
    red_mask = np.zeros(mask.shape).astype(np.float64)
    red_mask[np.where(s_mask>250)] = [0,0,255]
    alpha = 1
    beta = 0.6
    gamma = 0
    # print(red_mask.shape, img.shape)
    mask_img = cv2.addWeighted(new_img, alpha, red_mask, beta, gamma)
    return mask_img

def get_args():
    parser = argparse.ArgumentParser(description='generate mask',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img')
    parser.add_argument('--mask')
    return parser.parse_args()

def main():
    args = get_args()
    mask_files = sorted(listdir(args.mask))
    # img_files = listdir(args.img)
    for fn in mask_files:
        print(fn)
        mask_im = cv2.imread(join(args.mask, fn))
        img_im = cv2.imread(join(args.img, fn))
        blend = blend_mask(mask_im, img_im)
        cv2.imwrite(join(args.mask, fn), blend)

def add_mask(f_mask, f_img):
    # args = get_args()
    mask_files = sorted(listdir(f_mask))
    # img_files = listdir(args.img)
    for fn in mask_files:
        print(fn)
        mask_im = cv2.imread(join(f_mask, fn))
        img_im = cv2.imread(join(f_img, fn))
        blend = blend_mask(mask_im, img_im)
        cv2.imwrite(join(f_mask, fn), blend)


if __name__ == "__main__":
    main()

