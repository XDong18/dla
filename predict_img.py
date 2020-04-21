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
from add_mask import add_mask


def sample(img, rate, value):
    simg = img[:,:,0]
    h, w = simg.shape
    samples = np.random.random((h, w))
    noise_img_arg = np.where(samples < rate)
    noise_img = np.zeros((h,w))
    noise_img[noise_img_arg] = value
    simg[np.where(simg==0)] = noise_img[np.where(simg==0)]
    # noise_img = np.expand_dims(noise_img, 2)
    noise_img = np.stack((simg, simg, simg), 2)
    return noise_img

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path) 

def save_prob_images(prob, filenames, output_dir):


    fn = os.path.join(output_dir, filenames + '.png')
    out_dir = split(fn)[0]
    if not exists(out_dir):
        os.makedirs(out_dir)
    # im.save(fn)
    img = prob.squeeze().data.cpu().numpy() * 255
    print("prob", img.shape)
    # img = np.transpose(img,(1,2,0))
    cv2.imwrite(fn, img)

def save_output_images(predictions, filenames, output_dir):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    fn = os.path.join(output_dir, filenames + '.png')
    out_dir = split(fn)[0]
    if not exists(out_dir):
        os.makedirs(out_dir)
    predictions = np.transpose(predictions,(1,2,0))*255
    cv2.imwrite(fn, predictions)


def predict_img(model,
                full_img,
                name,
                output_dir):
    args = get_args()
    model.eval()
    img = full_img.resize((1920, 1216))
    img = np.array(img)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    if img.max()>1:
        img = img / 255.
    input_img = torch.from_numpy(img).cuda().float()
    with torch.no_grad():
        # image_var = Variable(input_img, requires_grad=False, volatile=True)
        final = model(input_img)[0]
        _, pred = torch.max(final, 1)
        pred[torch.where(pred==1)] = 0
        pred[torch.where(pred==13)] = 1
        pred[torch.where(pred==14)] = 1
        pred[torch.where(pred==15)] = 1
        pred[torch.where(pred!=1)] = 0
        pred = pred.cpu().data.numpy()

    save_output_images(pred, name, output_dir)



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    # parser.add_argument('--output', '-o', help='Filenames of ouput images')
    parser.add_argument('--out-dir', '-d', help='dir of ouput images')
    parser.add_argument('--arch')
    parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                        help='Downsampling ratio of IDA network output, which '
                             'is then upsampled to the original resolution '
                             'with bilinear interpolation.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('--input_dir', default='')
    parser.add_argument('--rotate', default=False, type=bool)
    parser.add_argument('--dense', default=False, type=bool)

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    if args.input_dir != '':
        in_files = sorted(listdir(args.input_dir))
    else:
        in_files = args.input

    single_model = dla_up.__dict__.get(args.arch)(
        args.classes, down_ratio=args.down)
    checkpoint = torch.load(args.resume)
    single_model.load_state_dict(checkpoint)
    # single_model.load_state_dict(checkpoint['state_dict'])
    model = torch.nn.DataParallel(single_model).cuda() #TODO

    # print(checkpoint['best_prec1'])

    print("Model loaded !")

    print(args.out_dir)
    print(args.input_dir)
    mkdir(args.out_dir)
    for i, fn in enumerate(in_files):
        print(fn.split('.')[0])
        logging.info("\nPredicting image {} ...".format(fn))
        fname = fn.split('.')[0]
        # fname = 'pred_000002_dense' #TODO

        if args.input_dir != '':
            # img = cv2.imread(join(args.input_dir,fn))
            img = Image.open(join(args.input_dir,fn))
            # print(fname)
        else:
            img = cv2.imread(fn)
            img = Image.open(fn)
            # print(fn, img.shape)

        if args.rotate==True:
            noise_img = sample(img, 0.0003, 254)
            img = noise_img
        
        predict_img(model=model,
                    full_img=img,
                    name=fname,
                    output_dir=args.out_dir)
    
    add_mask(args.out_dir, args.input_dir)
