import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# add by Bobby.
import copy
from training_functions2 import *
from resnet18 import *
from functools import partial
# end add.

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--pruning_fact', default=0.4, type=float, help="proportion of inference time that will approx be pruned away")
parser.add_argument('--perf_table', default='res-40-2', type=str, help='the perf_table (assumed to be in perf_tables '
                                                                       'folders and without the extension)')
parser.add_argument('--net', default='res', type=str, help='network architecture')
parser.add_argument('--init_red_fact', default=20, type=int,
                    help='fraction of the initial inference time that will be pruned at the first step, i.e. if network'
                         ' takes .1s to make a prediction, and init_red_fact = 20, the network will have an inference '
                         'time of approximately 19/20 * 0.1s after one pruning step')
parser.add_argument('--decay_rate', default=0.98, type=float, help='rate at which the target resource reduction for '
                                                                   'one step is reduced')
parser.add_argument('--holdout_prop', default=0.001, type=float, help='fraction of training set used for holdout')
parser.add_argument('--short_term_fine_tune', default=1, type=int, help='number of batches ')
parser.add_argument('--long_term_fine_tune', default=1, type=int, help='long term fine tune on the whole dataset, '
                                                                       'set to 0 by default because training from '
                                                                       'scratch gives better results')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--pruning_method', choices=['l2'], type=str, default='l2',
                    help='pruning algo to use, we tried fisher pruning as well in '
                         'https://github.com/NatGr/Master_Thesis/tree/master/NetAdapt but it does not show better '
                         'results and is much more complex, it was thus removed here')
parser.add_argument('--allow_small_prunings', action='store_true',
                    help="allows to prune from a layer even if it doesn't make us achieve the reduction objective")
parser.add_argument('--save_file', default='ResNet18', type=str, help='save file for checkpoints')
parser.add_argument('--width', default=1.0, type=float, metavar='D')
parser.add_argument('--depth', default=18, type=int, metavar='W')
#-------------------
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists('nbr_channels'):
    os.makedirs('nbr_channels')
args = parser.parse_args()

val_loader = main(parser)
model = torch.load('model.pth')

error_history = []
prune_history = []
table_costs_history = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ### validation code ###
    if args.evaluate:
        validate(val_loader, model, criterion, args)

    # Save
    #torch.save(model, 'model.pth')

