import argparse
import os
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
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
    if args.fast_test:
        args.batch_size = 1
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    model, train_loader, val_loader = main_worker(args.gpu, ngpus_per_node, args)
    return model, train_loader, val_loader

def validate(val_loader, model, criterion, args):
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
            if args.fast_test and (i >= 1):
                break
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
    batch_time = time.time()
    for i in range(no_steps): # no_steps is defined by args.short_term_fine_tune and args.long_term_fine_tune (unit: batch)
        if i%500 == 0 and i != 0:
            print(f"500 batches cost me {time.time()-batch_time} sec")
            batch_time = time.time()
        try:
            batch_in, batch_target = dataiter.next()
        except StopIteration:
            dataiter = iter(dataloader)
            batch_in, batch_target = dataiter.next()
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

## used to show pruning_record.
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

class AverageMeter(object): # similar to averagemeter 2
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

class AverageMeter2(object): # similar to averagemeter
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
