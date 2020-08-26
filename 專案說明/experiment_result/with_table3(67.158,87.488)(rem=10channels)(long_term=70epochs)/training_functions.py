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
import torch.optim
import torch.utils.data
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

    # create model
    if args.pretrained:
        print("=> using pre-trained model resnet18")
        model = resnet18(pretrained=True, progress=True)
        #model = resnet18()
        #checkpoint = torch.load(os.path.join('checkpoints', 'ResNet18.pth'),
        #                        map_location='cpu')['state_dict']
        #model.load_state_dict(checkpoint, strict=False)
    else:
        print("=> creating model resnet18")
        model = resnet18(pretrained=False, progress=True)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()

    ### define loss function (criterion) and optimizer ###
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    ### optionally resume from a checkpoint ###
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ### Data loading code ###
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    return model, train_loader, val_loader
    
    ### validation code ###
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

def main(parser):
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    model, train_loader, val_loader = main_worker(args.gpu, ngpus_per_node, args)
    return model, train_loader, val_loader

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
    n_batch = 1
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        #n_batch,
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            #if i > n_batch:
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
    print(f'lr={lr}')
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
    #process_time = time.time()
    batch_time = time.time()
    #print(f"no_steps:{no_steps}")
    for i in range(no_steps):
        if i%500 == 0 and i != 0:
            print(f"500 batches cost me {time.time()-batch_time} sec")
            batch_time = time.time()
        try:
            batch_in, batch_target = dataiter.next()
        except StopIteration:
            dataiter = iter(dataloader)
            batch_in, batch_target = dataiter.next()
        #print(f"batch_in_size = {batch_in.size()}")
        batch_in, batch_target = batch_in.to(device), batch_target.to(device)
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
