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
# from density import get_neg_map, get_dense, get_dense_map


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

    # im = Image.fromarray(
    #     (prob[1].squeeze().data.cpu().numpy() * 255).astype(np.uint8))
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
    # pdb.set_trace()
    # im = Image.fromarray(predictions.astype(np.uint8))
    # print(filenames, output_dir)
    fn = os.path.join(output_dir, filenames + '.png')
    out_dir = split(fn)[0]
    if not exists(out_dir):
        os.makedirs(out_dir)
    # im.save(fn)
    # print(predictions.shape)
    predictions = np.transpose(predictions,(1,2,0))*255
    cv2.imwrite(fn, predictions)


def predict_img(model,
                full_img,
                name,
                output_dir):
    args = get_args()
    model.eval()
    # cv2.imwrite('0_000000.jpg', np.array([BasicDataset.preprocess_image(full_img)]))
    # if args.dense:
    #     neg_mask = get_dense_mask(full_img, 4)
    with torch.no_grad():
        input_img = torch.from_numpy(np.array([BasicDataset.preprocess_image(full_img)])).cuda().float()
        # image_var = Variable(input_img, requires_grad=False, volatile=True)
        final = model(input_img)[0]
        # print(final.cpu().data)
        # print("final", final.shape)
        _, pred = torch.max(final, 1)

        # print("pred", pred.shape)
        pred = pred.cpu().data.numpy()

    save_output_images(pred, name, output_dir)



def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
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
    # parser.add_argument('--npy', default=True)

    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()
    if args.input_dir != '':
        in_files = listdir(args.input_dir)
    else:
        in_files = args.input

    single_model = dla_up.__dict__.get(args.arch)(
        args.classes, down_ratio=args.down)
    model = torch.nn.DataParallel(single_model).cuda()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print(checkpoint['best_prec1'])

    print("Model loaded !")

    print(args.out_dir)
    print(args.input_dir)
    mkdir(args.out_dir)
    for i, fn in enumerate(in_files):
        print(fn.split('.')[0])
        logging.info("\nPredicting image {} ...".format(fn))
        fname = fn.split('.')[0]
        if args.input_dir != '':
            img = cv2.imread(join(args.input_dir,fn))
            # print(fname)
        else:
            img = cv2.imread(fn)
            print(fn, img.shape)

        if args.rotate==True:
            noise_img = sample(img, 0.0003, 254)
            img = noise_img
            
        predict_img(model=model,
                    full_img=img,
                    name=fname,
                    output_dir=args.out_dir)
