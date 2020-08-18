import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
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
from training_functions import *
from resnet18 import *
from functools import partial
# end add.

parser = argparse.ArgumentParser(description='PyTorch ImageNet')
parser.add_argument('--training', default=False, help='True for training')
parser.add_argument('--pruning_fact', default=0.3, type=float, help="proportion of inference time that will approx be pruned away")
parser.add_argument('--perf_table', default='resnet18_table', type=str, help='the perf_table (assumed to be in perf_tables '
                                                                       'folders and without the extension)')
parser.add_argument('--net', default='res', type=str, help='network architecture')
parser.add_argument('--init_red_fact', default=10, type=int,
                    help='fraction of the initial inference time that will be pruned at the first step, i.e. if network'
                         ' takes .1s to make a prediction, and init_red_fact = 20, the network will have an inference '
                         'time of approximately 19/20 * 0.1s after one pruning step')
parser.add_argument('--decay_rate', default=0.98, type=float, help='rate at which the target resource reduction for '
                                                                   'one step is reduced')
parser.add_argument('--holdout_prop', default=0.9, type=float, help='fraction of training set used for holdout')
parser.add_argument('--short_term_fine_tune', default=1692, type=int, help='number of batches ')
parser.add_argument('--long_term_fine_tune', default=5078, type=int, help='long term fine tune on the whole dataset, '
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
#-------------------
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('-p', '--print-freq', default=500, type=int,
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


#pruning code

model, full_train_loader, val_loader = main(parser)
#model = torch.load('pruned_model10.pth')

if args.evaluate:
    model_list = []
    for i in range(1, 12):
        tmp = torch.load(f'pruned_model{i}.pth')
        model_list.append(tmp)
    


error_history = []
prune_history = []
table_costs_history = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.to(device)
model.load_table(os.path.join("perf_tables", f"{args.perf_table}.pickle"))

print('==> Preparing data..')

traindir = os.path.join(args.data, 'train')
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
# dataset split into train/holdout set
train_loader, holdout_loader = get_train_holdout(args.data, args.workers, args.batch_size, args.holdout_prop, args)

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ### validation code ###
    if args.evaluate:
        #for a_model in model_list:
            #validate(val_loader, a_model, criterion, args)
        validate(val_loader, model, criterion, args)
        exit()
    #prune_history.append(None)
    #table_costs_history.append(model.total_cost)
    #error_history.append(validate(val_loader, model, criterion, args))
    # validate on holdout so that we can compute the error change after having pruned one layer
    prev_holdout_error = validate(val_loader, model, criterion, args)
    red_objective = (1 - args.pruning_fact) * model.total_cost
    target_gains = args.pruning_fact * model.total_cost / args.init_red_fact  # gains at first epoch to achieve the
    # objective
    step_number = 1
    i = 0 # for saving model in pruning process
    while model.total_cost > red_objective:

        i = i + 1
        print(f"Pruning step number {step_number} -- target_gains are {target_gains/1000000 :.4f}s:")
        # Prune
        best_network, best_error, best_gains, pruned_layer, number_pruned = None, None, None, None, None

        # done in two steps to reduce number of memory transfers
        layer_mask_channels_gains = []
        for layer in model.to_prune:
            num_channels, gains = model.choose_num_channels(layer, target_gains, args.allow_small_prunings)
            if num_channels is not None:
                remaining_channels = model.choose_which_channels(layer, num_channels)
                layer_mask_channels_gains.append((layer, remaining_channels, num_channels, abs(gains)))
        model.cpu()  # stores it on CPU to avoid having 2 models on GPU at the same time
        
        for layer, remaining_channels, new_num_channels_pruned, new_gains in layer_mask_channels_gains:
            #print(f'Now pruning: "{layer}"')
            
            # creates a new model with the new mask to be fine_tuned
            new_model = copy.deepcopy(model)
            #print(f'old_model.perf_table:{model.perf_table}')
            #print(f'new_model.perf_table:{new_model.perf_table}')
            #new_model.perf_table = model.perf_table
            new_model.total_cost = model.total_cost - new_gains
            #new_model.num_channels_dict = model.num_channels_dict.copy() # necessary for rebuilding the network
            new_model.prune_channels(layer, remaining_channels)
            new_model.to(device)

            optimizer = torch.optim.SGD([v for v in new_model.parameters() if v.requires_grad],
                                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            finetune(new_model, optimizer, criterion, args.short_term_fine_tune, full_train_loader, layer, device)

            new_error = validate(val_loader, new_model, criterion, args)

            #delta_error = new_error - prev_holdout_error # orgin: negative number
            delta_error = prev_holdout_error - new_error # positive number
            relative_delta_error = delta_error / (new_gains/1000000)

            print(f"layer {layer} \t channels pruned {new_num_channels_pruned} \t error increase {delta_error :.2f} \t "
                  f"predicted gains {new_gains/1000000 :.4f} \t ratio {relative_delta_error :.2f}\n")

            prev_ratio = ((best_error - prev_holdout_error) / best_gains) if best_error is not None else 0

            # if we lose precision, best error_increase/cost_decrease ratio wins, otherwise,
            # better gain of precision wins
            if best_error is None or relative_delta_error < prev_ratio:
                new_model.cpu()
                best_network, best_error, pruned_layer = new_model, new_error, layer
                best_gains, number_pruned = new_gains, new_num_channels_pruned
            else:
                del new_model

        if best_network is None:
            raise Exception('We could not find a single layer to prune')
        print(f"the best validation error achieved was of {best_error} for layer {pruned_layer};"
              f" {number_pruned} channels were pruned; inference time gains of {best_gains/1000000 :.4f}s")
        torch.cuda.empty_cache()  # frees GPU memory to avoid running out of ram
        best_network.to(device)
        model = best_network
        ### create file to see # of remaining channels
        example = torch.rand(1, 3, 224, 224).cuda()
        traced_script_module = torch.jit.trace(model, example)
        traced_script_module.save(f"traced_model{i}.pt")
        torch.save(model, f'pruned_model{i}.pth')
        ###
        prev_holdout_error = best_error

        prune_history.append((pruned_layer, number_pruned))
        table_costs_history.append(model.total_cost)

        # evaluate on validation set
        #error_history.append(validate(model, val_loader, criterion, device, memory_leak=True))

        # prepare next step
        step_number += 1
        target_gains *= args.decay_rate
        

    print(f"pruned network inference time according to perf_table: {model.total_cost/1000000 :.4f}s")

    # long term fine tune
    if args.long_term_fine_tune != 0:
        optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        finetune(model, optimizer, criterion, args.long_term_fine_tune, full_train_loader, "", device)
        #error_history.append(validate(val_loader, model, criterion, args))
        prune_history.append(None)
        table_costs_history.append(table_costs_history[-1])
    print("finish pruning")
    # Save
    torch.save(model, 'pruned_model_final.pth')
    '''
    filename = os.path.join('checkpoints', f'{args.save_file}.pth')
    torch.save({
        'epoch': step_number,
        'state_dict': model.state_dict(),
        'error_history': error_history,
        'prune_history': prune_history,
        'table_costs_history': table_costs_history,
    }, filename)

    filename2 = os.path.join('nbr_channels', f'{args.save_file}.pickle')
    with open(filename2, 'wb') as file:
        pickle.dump(model.num_channels_dict, file)
    '''
