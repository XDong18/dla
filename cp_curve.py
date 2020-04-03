import torch
from segment import validate
from torch import nn
from ptc_dataset import BasicDataset
import dla_up
import argparse
from os import listdir
from torch.utils.tensorboard import SummaryWriter
from os.path import join


def parse_args():
    parser = argparse.ArgumentParser(
        description='IOU curve')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--down', default=2, type=int, choices=[2, 4, 8, 16],
                        help='Downsampling ratio of IDA network output, which '
                             'is then upsampled to the original resolution '
                             'with bilinear interpolation.')
    parser.add_argument('-j', '--workers', type=int, default=8)
    args = parser.parse_args()
    return args

def mIOU(output, target):
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
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

def my_val():
    writer = SummaryWriter(log_dir= "0308_iou")
    args = parse_args()
    val_dir_img = '/shared/xudongliu/data/argoverse-tracking/argo_track_aggre5/val/npy_img/'
    val_dir_mask = '/shared/xudongliu/data/argoverse-tracking/argo_track_aggre5/val/npy_mask/'
    my_val = BasicDataset(val_dir_img, val_dir_mask)
    val_loader = torch.utils.data.DataLoader(
        # SegList(data_dir, 'val', transforms.Compose([
        #     transforms.RandomCrop(crop_size),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     normalize,
        # ]),
        # binary=(args.classes == 2)),
        my_val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,pin_memory=True)
    
    files = listdir(args.resume)
    files.remove('model_best.pth.tar')
    files = sorted(files, key= lambda x:int(x.split('.')[0].split('_')[1]))
    for i, file in enumerate(files):
        single_model = dla_up.__dict__.get(args.arch)(
            args.classes, down_ratio=args.down)
        model = torch.nn.DataParallel(single_model).cuda()
        checkpoint = torch.load(join(args.resume, file))
        print(checkpoint['epoch'])
        model.load_state_dict(checkpoint['state_dict'])
        criterion = nn.NLLLoss2d(ignore_index=-1)
        score = validate(val_loader, model, criterion, eval_score=mIOU, print_freq=10)
        writer.add_scalar('IoU/epoch', score, i + 1)
        print(score)
    
    writer.close()


if __name__=="__main__":
    my_val()
# cp_path = "008cp"

