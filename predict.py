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

# def get_dense_mask(img, the):
#     img = img[:,:,0]
#     H, xedges, yedges = get_dense(img, 1920/40, 1200/40)
#     dense_map = get_dense_map(H, xedges, yedges)
#     neg_map = get_neg_map(dense_map, the)
#     return neg_map

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
    input_img = torch.from_numpy(np.array([BasicDataset.preprocess_image(full_img)])).cuda().float()
    image_var = Variable(input_img, requires_grad=False, volatile=True)
    final = model(image_var)[0]
    # print(final.cpu().data)
    # print("final", final.shape)
    _, pred = torch.max(final, 1)
    # print("pred", pred.shape)
    pred = pred.cpu().data.numpy()
    # final = final.cpu().data.numpy()
    # plt.imshow(final[0][0])
    # plt.colorbar()
    # plt.savefig(join(output_dir, 'img_0.png'))
    # plt.imshow(final[0][1])
    # plt.colorbar()
    # plt.savefig(join(output_dir, 'img_1.png'))
    # np.set_printoptions(threshold=2000*1300*2)
    # print(final)
    # batch_time.update(time.time() - end)
    # prob = torch.exp(final)
    # print("prob", prob.shape)
    save_output_images(pred, name, output_dir)
    # if prob.size(1) == 2:
    #     save_prob_images(prob, name, output_dir + '_prob')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--model', '-m', default='MODEL.pth',
    #                     metavar='FILE',
    #                     help="Specify the file in which the model is stored")
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
        # fname = 'pred_000002_dense' #TODO
        # fname = fn.split('.')[0] + '_noised_0.5_1'
        # if args.npy:
        #     print(join(args.input_dir,fn))
        #     img = np.load(join(args.input_dir,fn))
        #     print(img.shape)
        # else:
        # print("no npy")
        if args.input_dir != '':
            img = cv2.imread(join(args.input_dir,fn))
            # print(fname)
        else:
            img = cv2.imread(fn)
            print(fn, img.shape)

        if args.rotate==True:
            # print("rotate")
            # rows,cols,_ = img.shape
            # delta_y = -100
            # M=np.float32([[1,0,0],[0,1,delta_y]])
            # moved_img = cv2.warpAffine(img,M,(cols,rows))
            # img = np.zeros(img.shape)
            # noise_img = sample(img[:], 0.005, 1)
            # input_img1 = BasicDataset.preprocess_image(img)
            # input_img1 = np.transpose(input_img1, (1, 2, 0)) * 255.
            # input_img1 = np.concatenate((input_img1, input_img1, input_img1), axis=-1)
            # cv2.imwrite('test_img_no.png', input_img1)
            # noise_img = img
            noise_img = sample(img, 0.0003, 254)
            # angle = -10
            # rotated_img = scipy.ndimage.interpolation.rotate(img, angle, reshape=False)
            # fliped_img = cv2.flip(img, 0)
            # print(rotated_img.shape)
            # img = rotated_img
            # img = moved_img
            # input_img1 = BasicDataset.preprocess_image(img)
            # input_img1 = np.transpose(input_img1, (1, 2, 0)) * 255.
            # input_img1 = np.concatenate((input_img1, input_img1, input_img1), axis=-1)
            # input_img2 = BasicDataset.preprocess_image(noise_img)
            # input_img2 = np.transpose(input_img2, (1, 2, 0)) * 255.
            # input_img2 = np.concatenate((input_img2, input_img2, input_img2), axis=-1)
            # diff = (input_img2 - input_img1)
            # # diff = diff[np.where(diff!=0)]
            # cv2.imwrite('test_img_noise.png', input_img2)
            # cv2.imwrite('test_img_diff.png', diff)

            # print(diff)

            img = noise_img
            
            # img = fliped_img

            # print(join(args.out_dir, fname+'_'+str(angle)+'.png'))
            # cv2.imwrite(join(args.out_dir, fname+'_'+str(angle)+'.png'), rotated_img)
            # cv2.imwrite(join(args.out_dir, fname+'_'+'flip'+'.png'), fliped_img)
            # cv2.imwrite(join(args.out_dir, '000002'+'_'+'moved_'+str(delta_y)+'.png'), moved_img)
            # cv2.imwrite(join(args.out_dir, '000002'+'_'+'noised_'+"0.0045_255"+'.png'), noise_img)


            # cv2.imwrite('rotated.png', rotated_img)
        # img = Image.open(fn)
        predict_img(model=model,
                    full_img=img,
                    name=fname,
                    output_dir=args.out_dir)
