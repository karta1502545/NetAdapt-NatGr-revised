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

import torch.nn.functional as torch_fct
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

#from pruning_functions import *
from resnet18 import *

best_acc1 = 0

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    cudnn.benchmark = True

    ### Data loading code ###
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return val_loader

def main(parser):
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        
    else:
        # Simply call main_worker function
        val_loader = main_worker(args.gpu, ngpus_per_node, args)
        return val_loader

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        
        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    #print(val_loader)
    #n_batch = 1
    print('from training_functions.py line265:')
    #print(f'n_batch={n_batch}')
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            #if i >= n_batch:
                #break
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            #print(images[0].type())
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class TensorDataSetWithAugmentation(Dataset):
    """subclass of Dataset than handles data augmentation for tensors"""
    def __init__(self, x_tensor, y_tensor, transform=None):
        super(TensorDataSetWithAugmentation, self).__init__()
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transform = transform

    def __getitem__(self, item):
        x = self.x_tensor[item, :, :, :]
        y = self.y_tensor[item]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return self.x_tensor.size(0)
'''
def get_train_holdout(data_loc, workers, batch_size, holdout_prop, args):
    """get train and holdout set, to perform pruning as in the NetAdapt paper
    :param data_loc: the location of the cifar dataset on disk
    :param workers: the number of workers the data loaders have to use
    :param batch_size: the batch size to use for training
    :param holdout_prop: fraction of the total dataset that will end up in holdout set
    :return (train_loader, holdout_loader): the train set and holdout set loaders"""
    traindir = os.path.join(args.data, 'train')
    print('1')
    tmp = datasets.ImageFolder(traindir,
    transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]))
    print('2')
    size = tmp.__len__()
    full_train = np.zeros((size, 3, 224, 224), dtype=np.float32)
    full_train_labels = np.zeros(size, dtype=np.long)
    print('2.5')
    print(size)
    for i in range(size):
        img_and_label = tmp.__getitem__(i)
        full_train[i, :, :, :] = img_and_label[0].numpy()
        full_train_labels[i] = img_and_label[1]
    print('3')
    x_train, x_holdout, y_train, y_holdout = train_test_split(full_train, full_train_labels, test_size=holdout_prop,
                                                              stratify=full_train_labels, random_state=17)  # so as to
    # always have the same separation since it's used in several files
    print('4')
    train_mean = np.mean(x_train, axis=(0, 2, 3))  # so that we don't use the mean/std of the holdout set
    train_std = np.std(x_train, axis=(0, 2, 3))
    print('5')
    norm_transform = transforms.Normalize(train_mean, train_std)
    transform_train = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    # from numpy array to tensor
    x_train, x_holdout = torch.from_numpy(x_train), torch.from_numpy(x_holdout)
    y_train, y_holdout = torch.from_numpy(y_train), torch.from_numpy(y_holdout)

    train_set = TensorDataSetWithAugmentation(x_train, y_train, transform=transform_train)
    holdout_set = TensorDataSetWithAugmentation(x_holdout, y_holdout, transform=norm_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
    holdout_loader = DataLoader(holdout_set, batch_size=50, shuffle=False, num_workers=workers, pin_memory=False)

    return train_loader, holdout_loader
'''
def get_train_holdout(data_loc, workers, batch_size, holdout_prop, args):
    """get train and holdout set, to perform pruning as in the NetAdapt paper
    :param data_loc: the location of the cifar dataset on disk
    :param workers: the number of workers the data loaders have to use
    :param batch_size: the batch size to use for training
    :param holdout_prop: fraction of the total dataset that will end up in holdout set
    :return (train_loader, holdout_loader): the train set and holdout set loaders"""
    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(traindir,
        transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
        ]))
    #print(len(train_dataset))
    #print(holdout_prop)
    train_set, holdout_set = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*(1-holdout_prop)), (len(train_dataset)-int(len(train_dataset)*(1-holdout_prop)))])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=False)
    holdout_loader = DataLoader(holdout_set, batch_size=200, shuffle=False, num_workers=workers, pin_memory=False)

    return train_loader, holdout_loader

def finetune(model, optimizer, criterion, no_steps, dataloader, layer, device):
    """fine tunes the given model
    :param model: the model to fine tune
    :param optimizer: the optimizer to use to optimize the model
    :param criterion: the criterion to compute the loss
    :param no_steps: the number of steps to fine tune
    :param dataloader: the pytoch dataloader to get the training samples from
    :param device: the pytorch device to perform computations on
    :param layer: the name of the layer that was just pruned
    """
    model.train()  # switch to train mode
    begin = time.time()
    dataiter = iter(dataloader)
    losses = AverageMeter2()
    errors = AverageMeter2()
    print('1')
    for i in range(no_steps):
        print('2')
        try:
            batch_in, batch_target = dataiter.next()
        except StopIteration:
            dataiter = iter(dataloader)
            batch_in, batch_target = dataiter.next()
        print('3')
        batch_in, batch_target = batch_in.to(device), batch_target.to(device)
        print('4')
        # compute output
        output = model(batch_in)

        loss = criterion(output, batch_target)

        # measure accuracy and record loss
        error = get_error(output, batch_target)

        losses.update(loss.item(), batch_in.size(0))
        errors.update(error.item(), batch_in.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f' * fine tune after pruning {layer} * Elapsed seconds {time.time() - begin :.1f} \t '
          f'Loss {losses.avg :.3f} \t Error {errors.avg :.3f}')

def get_error(output, target):
    """given the output of the NN and the target labels, return the error"""
    _, output = output.max(1)  # output is one-hot encoded
    output = output.view(-1)
    return 100 - output.eq(target).float().sum().mul(100 / target.size(0))
    
class AverageMeter2(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
