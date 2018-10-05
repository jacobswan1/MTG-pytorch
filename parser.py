'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function

import torch
import argparse
import torch.optim as optim


def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch Attribute Grouning Training')
    parser.add_argument('--msg', default=False, type=bool, help='display message')
    parser.add_argument('--use_gpu', default=torch.cuda.is_available(), type=bool, help='Use GPU or not')
    parser.add_argument('--multi_gpu', default=(torch.cuda.device_count() > 0), type=bool, help='Use multi-GPU or not')
    parser.add_argument('--gpu_id', default=-1, type=int, help='Use specific GPU.')

    parser.add_argument('--optimizer', default=optim.SGD, help='optimizer')
    parser.add_argument('--num_workers', default=2, type=int, help='num of fetching threads')
    parser.add_argument('--batch_size', default=12, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--result_path', default='./results', help='result path')

    # Define the training parameters
    parser.add_argument('--class_num', default=5, type=int, help='num of fetchi  ng threads')
    parser.add_argument('--checkpoint_epoch', default=2, type=int, help='epochs to save checkpoint ')
    parser.add_argument('--lr_adjust_epoch', default=5, type=int, help='lr adjust epoch')
    parser.add_argument('--n_epoch', default=1000, type=int, help='training epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')

    # Define the checkpoint reloading path
    parser.add_argument('--resume', default='', help='result path')

    # Define the data_set path
    parser.add_argument('--img_path', default='/media/drive1/Data/coco17/train2017/', help='coco_train_2017 path')
    parser.add_argument('--annotation', default='/media/drive1/Data/coco17/annotations/'
                                                'captions_train2017.json', help='coco_train_2017 annotation path')
    parser.add_argument('--dictionary', default='./others/low-level-attr.txt', help='dict of attributes')
    args = parser.parse_args()

    return args

