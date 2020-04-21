import argparse
import json
import logging
import os
import threading
from os.path import exists, join, split, dirname

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
import torch.utils.data
from torch import nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import dla_up
import data_transforms as transforms
import dataset

from torch.utils.tensorboard import SummaryWriter
from ptc_dataset import BasicDataset
from dataset_trans import BasicDataset #TODO
from dataset_coco import COCOSeg
import copy
from torch.utils.data import DataLoader, random_split
import sys
sys.path.append('lib/')

try:
    from modules import batchnormsync
    HAS_BN_SYNC = True
except ImportError:
    HAS_BN_SYNC = False

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

global_step = 0  #TODO

CITYSCAPE_PALLETE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


def f_print(*obj):
    fn='./output.log'
    print(obj)
    with open(fn, 'a+') as f:
        print(obj, file=f)


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None,
                 out_name=False, out_size=False, binary=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.out_size = out_size
        self.binary = binary
        self.read_lists()

    def __getitem__(self, index):
        image = Image.open(join(self.data_dir, self.image_list[index]))
        data = [image]
        if self.label_list is not None:
            label_map = Image.open(join(self.data_dir, self.label_list[index]))
            if self.binary:
                label_map = Image.fromarray(
                    (np.array(label_map) > 0).astype(np.uint8))
            data.append(label_map)
        if self.bbox_list is not None:
            data.append(Image.open(join(self.data_dir, self.bbox_list[index])))
        data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        if self.out_size:
            data.append(torch.from_numpy(np.array(image.size, dtype=int)))
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        bbox_path = join(self.list_dir, self.phase + '_bboxes.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)
        if exists(bbox_path):
            self.bbox_list = [line.strip() for line in open(bbox_path, 'r')]
            assert len(self.image_list) == len(self.bbox_list)


class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir,
                                        self.label_list[index])))
        # data = list(self.transforms(*data))
        if len(data) > 1:
            out_data = list(self.transforms(*data))
        else:
            out_data = [self.transforms(*data)]
        ms_images = [self.transforms(data[0].resize((int(w * s), int(h * s)),
                                                    Image.BICUBIC))
                     for s in self.scales]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def validate(val_loader, model, criterion, eval_score=None, print_freq=10, cityscape=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()
    pos_scores = AverageMeter()
    neg_scores = AverageMeter()
    recall_scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():#TODO
        for i, (input, target) in enumerate(val_loader):
            if type(criterion) in [torch.nn.modules.loss.L1Loss,
                                torch.nn.modules.loss.MSELoss]:
                target = target.float()
            input = input.cuda().float()
            # target = target.cuda(async=True)
            target = target.cuda().long()
            output = model(input)[0]
            # print("model done") 
            # print(output)
            loss = criterion(output, target)
            # print("loss done")


            # measure accuracy and record loss
            # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            # losses.update(loss.data[0], input.size(0))
            losses.update(loss.item(), input.size(0))
            # print("model done")
            if eval_score is not None:
                # TODO cahnged
                score.update(eval_score(output, target), input.size(0))
                recall_scores.update(recall(output, target), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Score {score.val:.3f} ({score.avg:.3f})\t'
                    'recall_score {recall_score.val:.3f} ({recall_score.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    score=score, recall_score=recall_scores)) #TODO , flush=True
    
    print(' * Score {top1.avg:.3f}\recall_score{recall_score.avg:.3f}'.format(top1=score, recall_score=recall_scores))

    return score.avg, losses.avg, recall_scores.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    # correct = correct[target != 255]
    correct = correct[target != 255]

    correct = correct.view(-1)
    if correct.size(0)!=0:
        score = correct.float().sum(0).mul(100.0 / correct.size(0))
    else:
        return 100.
    # return score.data[0]
    return score.item()


def recall(output, t_target):
    """Computes the precision@k for the specified values of k"""
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = copy.deepcopy(t_target)
    target[torch.where(target==0)] = 255 #Ignore background

    target = target.view(1, -1)
    correct = pred.eq(target)
    # correct = correct[target != 255]
    correct = correct[target != 255]

    correct = correct.view(-1)
    if correct.size(0)!=0:
        score = correct.float().sum(0).mul(100.0 / correct.size(0))
    else:
        return 100.
    # return score.data[0]
    return score.item()

def mIOU(output, target, cityscape=False):
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    if cityscape:
        pred[torch.where(pred==13)] = 1
        pred[torch.where(pred==14)] = 1
        pred[torch.where(pred==15)] = 1
        pred[torch.where(pred!=1)] = 0
        
    positive_target = target[target == 1]
    positive_pred = pred[target == 1]
    negtive_target = target[target == 0]
    negtive_pred = pred[target == 0]

    positive_union = positive_pred.eq(positive_target)
    positive_union = positive_union.view(-1).float().sum(0)

    positive_target = target[target != -1]
    positive_pred = pred[target != -1]
    pos_section_pred = positive_pred.eq(1).view(-1).float().sum(0)
    pos_section_target = positive_target.eq(1).view(-1).float().sum(0)
    pos_intersection = pos_section_pred + pos_section_target - positive_union

    if pos_intersection>0:
        pos_score = positive_union.mul(100.0 / pos_intersection).item()
    else:
        pos_score = 100.0

    negtive_union = negtive_pred.eq(negtive_target)
    negtive_union = negtive_union.view(-1).float().sum(0)

    negtive_target = target[target != -1]
    negtive_pred = pred[target != -1]
    neg_section_pred = negtive_pred.eq(0).view(-1).float().sum(0)
    neg_section_target = negtive_target.eq(0).view(-1).float().sum(0)
    neg_intersection = neg_section_pred + neg_section_target - negtive_union

    if neg_intersection>0:
        neg_score = negtive_union.mul(100.0 / neg_intersection).item()
    else:
        neg_score = 100.0
    #print("pos", pos_score, "neg", neg_score)
    return (pos_score + neg_score) / 2

def posIOU(output, target, cityscape=False):
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    if cityscape:
        pred[torch.where(pred==13)] = 1
        pred[torch.where(pred==14)] = 1
        pred[torch.where(pred==15)] = 1
        pred[torch.where(pred!=1)] = 0
    positive_target = target[target == 1]
    positive_pred = pred[target == 1]
    # negtive_target = target[target == 0]
    # negtive_pred = pred[target == 0]

    positive_union = positive_pred.eq(positive_target)
    positive_union = positive_union.view(-1).float().sum(0)

    positive_target = target[target != -1]
    positive_pred = pred[target != -1]
    pos_section_pred = positive_pred.eq(1).view(-1).float().sum(0)
    pos_section_target = positive_target.eq(1).view(-1).float().sum(0)
    pos_intersection = pos_section_pred + pos_section_target - positive_union

    if pos_intersection>0:
        pos_score = positive_union.mul(100.0 / pos_intersection).item()
    else:
        pos_score = 100.0

    # negtive_union = negtive_pred.eq(negtive_target)
    # negtive_union = negtive_union.view(-1).float().sum(0)

    # negtive_target = target[target != -1]
    # negtive_pred = pred[target != -1]
    # neg_section_pred = negtive_pred.eq(0).view(-1).float().sum(0)
    # neg_section_target = negtive_target.eq(0).view(-1).float().sum(0)
    # neg_intersection = neg_section_pred + neg_section_target - negtive_union

    # if neg_intersection>0:
    #     neg_score = negtive_union.mul(100.0 / neg_intersection).item()
    # else:
    #     neg_score = 100.0
    #print("pos", pos_score, "neg", neg_score)
    return pos_score

def negIOU(output, target, cityscape=False):
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    if cityscape:
        pred[torch.where(pred==13)] = 1
        pred[torch.where(pred==14)] = 1
        pred[torch.where(pred==15)] = 1
        pred[torch.where(pred!=1)] = 0
    # positive_target = target[target == 1]
    # positive_pred = pred[target == 1]
    negtive_target = target[target == 0]
    negtive_pred = pred[target == 0]

    # positive_union = positive_pred.eq(positive_target)
    # positive_union = positive_union.view(-1).float().sum(0)

    # positive_target = target[target != -1]
    # positive_pred = pred[target != -1]
    # pos_section_pred = positive_pred.eq(1).view(-1).float().sum(0)
    # pos_section_target = positive_target.eq(1).view(-1).float().sum(0)
    # pos_intersection = pos_section_pred + pos_section_target - positive_union

    # if pos_intersection>0:
    #     pos_score = positive_union.mul(100.0 / pos_intersection).item()
    # else:
    #     pos_score = 100.0

    negtive_union = negtive_pred.eq(negtive_target)
    negtive_union = negtive_union.view(-1).float().sum(0)

    negtive_target = target[target != -1]
    negtive_pred = pred[target != -1]
    neg_section_pred = negtive_pred.eq(0).view(-1).float().sum(0)
    neg_section_target = negtive_target.eq(0).view(-1).float().sum(0)
    neg_intersection = neg_section_pred + neg_section_target - negtive_union

    if neg_intersection>0:
        neg_score = negtive_union.mul(100.0 / neg_intersection).item()
    else:
        neg_score = 100.0
    #print("pos", pos_score, "neg", neg_score)
    return neg_score

def train(train_loader, model, criterion, optimizer, epoch, lr_scheduler, writer, 
          eval_score=None, print_freq=10):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    pos_scores = AverageMeter()
    neg_scores = AverageMeter()
    scores = AverageMeter()
    recall_score = AverageMeter()

    global global_step
    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        
        data_time.update(time.time() - end)

        # pdb.set_trace()

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda().float()
        # target = target.cuda(async=True)
        target = target.cuda().long()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)[0]
        loss = criterion(output, target_var)
        writer.add_scalar('Loss/train', loss.item(), global_step)
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], input.size(0))
        losses.update(loss.item(), input.size(0))
        if eval_score is not None:
            scores.update(eval_score(output, target_var), input.size(0))
            recall_score.update(recall(output, target_var), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step() #TODO

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        global_step += 1
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Score {top1.val:.3f} ({top1.avg:.3f})\t'
                  'recall_Score {recall_score.val:.3f} ({recall_score.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=scores, recall_score=recall_score))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    args = parse_args()
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(args.checkpoint_dir, 'model_best.pth.tar'))


def train_seg(args):
    writer = SummaryWriter(comment = args.log)
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size
    checkpoint_dir = args.checkpoint_dir

    print(' '.join(sys.argv))
    # logger.info(' '.join(sys.argv))


    for k, v in args.__dict__.items():
        print(k, ':', v)

    pretrained_base = args.pretrained_base
    # print(dla_up.__dict__.get(args.arch))
    single_model = dla_up.__dict__.get(args.arch)(
        classes=args.classes, down_ratio=args.down)
    model = torch.nn.DataParallel(single_model).cuda()
    print('model_created')
    if args.bg_weight > 0:
        weight_array = np.ones(args.classes, dtype=np.float32)
        weight_array[0] = args.bg_weight
        weight = torch.from_numpy(
            weight_array)
        # criterion = nn.NLLLoss2d(ignore_index=255, weight=weight)
        criterion = nn.NLLLoss2d(ignore_index=255, weight=weight)
    else:
        # criterion = nn.NLLLoss2d(ignore_index=255)
        criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()


    t = []
    if args.random_rotate > 0:
        t.append(transforms.RandomRotate(args.random_rotate))
    if args.random_scale > 0:
        t.append(transforms.RandomScale(args.random_scale))
    t.append(transforms.RandomCrop(crop_size)) #TODO
    if args.random_color:
        t.append(transforms.RandomJitter(0.4, 0.4, 0.4))
    t.extend([transforms.RandomHorizontalFlip()]) #TODO

    t_val = []
    t_val.append(transforms.RandomCrop(crop_size))

    train_json = '/shared/xudongliu/COCO/annotation2017/annotations/instances_train2017.json'
    train_root = '/shared/xudongliu/COCO/train2017/train2017'
    my_train = COCOSeg(train_root, train_json, transforms.Compose(t), is_train=True)
    
    val_json = '/shared/xudongliu/COCO/annotation2017/annotations/instances_val2017.json'
    val_root = '/shared/xudongliu/COCO/2017val/val2017'
    my_val = COCOSeg(val_root, val_json, transforms.Compose(t_val), is_train=True)

  

    train_loader = torch.utils.data.DataLoader(
        my_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        my_val, batch_size=20, shuffle=False, num_workers=num_workers,pin_memory=True) #TODO  batch_size
    print("loader created")
    
    
    # optimizer = torch.optim.Adam(single_model.optim_parameters(),
    #                             args.lr,
    #                              weight_decay=args.weight_decay) #TODO adam optimizer
    optimizer = torch.optim.SGD(single_model.optim_parameters(),
                                args.lr,momentum=args.momentum,
                                 weight_decay=args.weight_decay) #TODO adam optimizer
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32) #TODO
    cudnn.benchmark = True
    best_prec1 = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion, eval_score=accuracy)
        return

    # TODO test val
    # print("test val")
    # prec1 = validate(val_loader, model, criterion, eval_score=accuracy)

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        print('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
        # train for one epoch
        
        train(train_loader, model, criterion, optimizer, epoch, lr_scheduler,
              eval_score=accuracy, writer=writer)

        checkpoint_path = os.path.join(checkpoint_dir,'checkpoint_{}.pth.tar'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict()
        }, is_best=False, filename=checkpoint_path)

        # evaluate on validation set
        prec1, loss_val, recall_val = validate(val_loader, model, criterion, eval_score=accuracy)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        writer.add_scalar('accuracy/epoch', prec1, epoch+1)
        writer.add_scalar('loss/epoch', loss_val, epoch+1)
        writer.add_scalar('recall/epoch', recall_val, epoch+1)

        checkpoint_path = os.path.join(checkpoint_dir,'checkpoint_{}.pth.tar'.format(epoch))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)

        if (epoch + 1) % args.save_freq == 0:
            history_path = 'checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)
    
    writer.close()


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10
    every 30 epochs"""
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def crop_image(image, size):
    left = (image.size[0] - size[0]) // 2
    upper = (image.size[1] - size[1]) // 2
    right = left + size[0]
    lower = upper + size[1]
    return image.crop((left, upper, right, lower))


def save_output_images(predictions, filenames, output_dir, sizes=None):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        if sizes is not None:
            im = crop_image(im, sizes[ind])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_prob_images(prob, filenames, output_dir, sizes=None):
    for ind in range(len(filenames)):
        im = Image.fromarray(
            (prob[ind][1].squeeze().data.cpu().numpy() * 255).astype(np.uint8))
        if sizes is not None:
            im = crop_image(im, sizes[ind])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def save_colorful_images(predictions, filenames, output_dir, palettes):
    """
    Saves a given (B x C x H x W) into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    for ind in range(len(filenames)):
        im = Image.fromarray(palettes[predictions[ind].squeeze()])
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)


def test(eval_data_loader, model, num_classes,
         output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    for iter, (image, label, name, size) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)
        final = model(image_var)[0]
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        prob = torch.exp(final)
        if save_vis:
            save_output_images(pred, name, output_dir, size)
            if prob.size(1) == 2:
                save_prob_images(prob, name, output_dir + '_prob', size)
            else:
                save_colorful_images(pred, name, output_dir + '_color',
                                     CITYSCAPE_PALLETE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            print('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        print('Eval: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              .format(iter, len(eval_data_loader), batch_time=batch_time,
                      data_time=data_time))
    ious = per_class_iu(hist) * 100
    print(' '.join('{:.03f}'.format(i) for i in ious))
    if has_gt:  # val
        return round(np.nanmean(ious), 2)


def resize_4d_tensor(tensor, width, height):
    tensor_cpu = tensor.cpu().numpy()
    if tensor.size(2) == height and tensor.size(3) == width:
        return tensor_cpu
    out_size = (tensor.size(0), tensor.size(1), height, width)
    out = np.empty(out_size, dtype=np.float32)

    def resize_one(i, j):
        out[i, j] = np.array(
            Image.fromarray(tensor_cpu[i, j]).resize(
                (width, height), Image.BILINEAR))

    def resize_channel(j):
        for i in range(tensor.size(0)):
            out[i, j] = np.array(
                Image.fromarray(tensor_cpu[i, j]).resize(
                    (width, height), Image.BILINEAR))

    workers = [threading.Thread(target=resize_channel, args=(j,))
               for j in range(tensor.size(1))]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return out


def test_ms(eval_data_loader, model, num_classes, scales,
            output_dir='pred', has_gt=True, save_vis=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    num_scales = len(scales)
    for iter, input_data in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        if has_gt:
            name = input_data[2]
            label = input_data[1]
        else:
            name = input_data[1]
        h, w = input_data[0].size()[2:4]
        images = [input_data[0]]
        images.extend(input_data[-num_scales:])
        outputs = []
        for image in images:
            image_var = Variable(image, requires_grad=False, volatile=True)
            final = model(image_var)[0]
            outputs.append(final.data)
        final = sum([resize_4d_tensor(out, w, h) for out in outputs])
        pred = final.argmax(axis=1)
        batch_time.update(time.time() - end)
        if save_vis:
            save_output_images(pred, name, output_dir)
            save_colorful_images(pred, name, output_dir + '_color',
                                 CITYSCAPE_PALLETE)
        if has_gt:
            label = label.numpy()
            hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
            logger.info('===> mAP {mAP:.3f}'.format(
                mAP=round(np.nanmean(per_class_iu(hist)) * 100, 2)))
        end = time.time()
        logger.info('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    if has_gt:  # val
        ious = per_class_iu(hist) * 100
        logger.info(' '.join('{:.03f}'.format(i) for i in ious))
        return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = dla_up.__dict__.get(args.arch)(
        args.classes, down_ratio=args.down)

    model = torch.nn.DataParallel(single_model).cuda()

    # data_dir = args.data_dir
    # info = dataset.load_dataset_info(data_dir)
    # normalize = transforms.Normalize(mean=info.mean, std=info.std)
    # scales = [0.5, 0.75, 1.25, 1.5, 1.75]
    # scales = [0.5, 0.75, 1.25, 1.5]
    # t = []
    # if args.crop_size > 0:
    #     t.append(transforms.PadToSize(args.crop_size))
    # t.extend([transforms.ToTensor(), normalize])
    # if args.ms:
    #     data = SegListMS(data_dir, phase, transforms.Compose(t), scales)
    # else:
    #     data = SegList(data_dir, phase, transforms.Compose(t),
    #                    out_name=True, out_size=True,
    #                    binary=args.classes == 2)

    
    test_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=False
    )

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = '{}_{:03d}_{}'.format(args.arch, start_epoch, phase)
    if len(args.test_suffix) > 0:
        out_dir += '_' + args.test_suffix

    if args.ms:
        out_dir += '_ms'

    if args.ms:
        mAP = test_ms(test_loader, model, args.classes, save_vis=True,
                      has_gt=phase != 'test' or args.with_gt,
                      output_dir=out_dir,
                      scales=scales)
    else:
        mAP = test(test_loader, model, args.classes, save_vis=True,
                   has_gt=phase != 'test' or args.with_gt, output_dir=out_dir)
    print('mAP: ', mAP)


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(
        description='DLA Segmentation and Boundary Prediction')
    parser.add_argument('cmd', choices=['train', 'test'])
    # parser.add_argument('-d', '--data-dir', default=None)
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--train-samples', default=16000, type=int)
    parser.add_argument('--loss', default='l1', type=str)
    # parser.add_argument('--test-batch-size', type=int, default=1000,
    #                     metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='- seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging '
                             'training status')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-base', default=None,
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                        help='Downsampling ratio of IDA network output, which '
                             'is then upsampled to the original resolution '
                             'with bilinear interpolation.')
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--lr-mode', default='step')
    parser.add_argument('--bn-sync', action='store_true', default=False)
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    parser.add_argument('--random-color', action='store_true', default=False)
    parser.add_argument('--save-freq', default=10, type=int)
    parser.add_argument('--ms', action='store_true', default=False)
    parser.add_argument('--bg-weight', type=float, default=-1)
    parser.add_argument('--test-suffix', default='')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-i', '--checkpoint-dir')   
    parser.add_argument('--log')        
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    return args


def main():
    args = parse_args()
    if not exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if args.bn_sync:
        if HAS_BN_SYNC:
            dla_up.set_bn(batchnormsync.BatchNormSync)
        else:
            print('batch normalization synchronization across GPUs '
                  'is not imported.')
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)

# validate(val_loader, model, criterion, eval_score=None, print_freq=10)

def my_val():
    args = parse_args()
    val_dir_img = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/image_02/'
    val_dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track/val/npy_mask/'
    my_val = BasicDataset(val_dir_img, val_dir_mask, None, is_train=False)
    val_loader = torch.utils.data.DataLoader(
        # SegList(data_dir, 'val', transforms.Compose([
        #     transforms.RandomCrop(crop_size),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ]),
        # binary=(args.classes == 2)),
        my_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=True)

    single_model = dla_up.__dict__.get(args.arch)(
        args.classes, down_ratio=args.down)
    checkpoint = torch.load(args.resume)
    # print(checkpoint['epoch'])
    # model.load_state_dict(checkpoint['state_dict'])
    single_model.load_state_dict(checkpoint)
    model = torch.nn.DataParallel(single_model).cuda()
    
    
    criterion = nn.NLLLoss2d(ignore_index=255)

    score = validate(val_loader, model, criterion, eval_score=accuracy, print_freq=10, cityscape=True)
    print(score)

if __name__ == '__main__':
    main()
    # my_val()
