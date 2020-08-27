import argparse
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets


# add by Bobby.
import copy
from pandas import DataFrame
from training_functions import *
from resnet18 import *
from functools import partial
# end add.

parser = argparse.ArgumentParser(description='PyTorch ImageNet')
parser.add_argument('--record', type=str, default=True, help='True for using excel file to record pruning process')
parser.add_argument('--training', type=bool, default=False, help='True for training more epochs after pruning')
parser.add_argument('--fast_test', type=bool, default=False, help='True for speed up test if there are some compile errors'
                                                       'by reducing the number(=1) of validation batch and batch size(=1).')
parser.add_argument('--pruning_fact', default=0.3, type=float, help="proportion of inference time that will approx be pruned away")
parser.add_argument('--perf_table', default='resnet18_table', type=str, help='the perf_table (assumed to be in perf_tables '
                                                                       'folders and without the extension)')
parser.add_argument('--training_model', default='pruned_model_final.pth', type=str)
parser.add_argument('--init_red_fact', default=10, type=int,
                    help='fraction of the initial inference time that will be pruned at the first step, i.e. if network'
                         ' takes .1s to make a prediction, and init_red_fact = 20, the network will have an inference '
                         'time of approximately 19/20 * 0.1s after one pruning step')
parser.add_argument('--decay_rate', default=0.98, type=float, help='rate at which the target resource reduction for '
                                                                   'one step is reduced')
parser.add_argument('--short_term_fine_tune', default=1692, type=int, help='number of batches, 1692*N->1/3*N epochs')
parser.add_argument('--long_term_fine_tune', default=5078, type=int, help='long term fine tune on the whole dataset, '
                                                                       '5078*N->N epochs')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--pruning_method', choices=['l2'], type=str, default='l2',
                    help='pruning algo to use, we tried fisher pruning as well in '
                         'https://github.com/NatGr/Master_Thesis/tree/master/NetAdapt but it does not show better '
                         'results and is much more complex, it was thus removed here')
parser.add_argument('--allow_small_prunings', action='store_true',
                    help="allows to prune from a layer even if it doesn't make us achieve the reduction objective")
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('-p', '--print-freq', default=500, type=int,
                    metavar='N', help='print frequency (default: 500)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default=True, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
                        
data={
    'acc1':[],
    'pruned_layer_name':[],
    'num_of_pruned_channels':[],
    'target_gains':[],
    'gains':[],
    'rem_total_param':[],
    'rem_trainable_param':[],
}


args = parser.parse_args()
print(args.record)

if args.fast_test:
    args.batch_size = 1

model, full_train_loader, val_loader = main(parser)
if args.training: # reload pruned model and train it.
    print(f"=> Now using {args.training_model}")
    model = torch.load(args.training_model)
    args.pruning_fact = 0

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model.to(device)
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

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ### validation code ###
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        exit()
    #######################
    
    # validate on validation dataset so that we can compute the accuracy changes after having pruned one layer
    prev_acc = validate(val_loader, model, criterion, args)
    
    if args.record == 'True':
        ### record initial model performance###
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data['pruned_layer_name'].append('None')
        data['acc1'].append(prev_acc.cpu().numpy())
        data['target_gains'].append(0)
        data['gains'].append(0)
        data['num_of_pruned_channels'].append(0)
        data['rem_total_param'].append(total_params)
        data['rem_trainable_param'].append(total_trainable_params)
        ########################################
    
    red_objective = (1 - args.pruning_fact) * model.total_cost # in the below while loop, model will be pruned until meet the reduction objective.
    target_gains = args.pruning_fact * model.total_cost / args.init_red_fact  # latency reduction constraint(sec) per while loop iteration 
    step_number = 1
    i = 0 # for saving model in pruning process

    while model.total_cost > red_objective:
        i = i + 1
        print(f"Pruning step number {step_number} -- target_gains are {target_gains/1000000 :.4f}s:")
        # Prune
        best_network, best_acc, best_gains, pruned_layer, number_pruned = None, None, None, None, None

        # done in two steps to reduce number of memory transfers
        layer_mask_channels_gains = []
        for layer in model.to_prune:
            num_channels, gains = model.choose_num_channels(layer, target_gains, args.allow_small_prunings)
            if num_channels is not None:
                remaining_channels = model.choose_which_channels(layer, num_channels)
                layer_mask_channels_gains.append((layer, remaining_channels, num_channels, abs(gains)))
        model.cpu()  # stores it on CPU to avoid having 2 models on GPU at the same time
        
        for layer, remaining_channels, new_num_channels_pruned, new_gains in layer_mask_channels_gains:
            
            # creates a new model with the new mask to be fine_tuned
            new_model = copy.deepcopy(model)
            new_model.total_cost = model.total_cost - new_gains
            new_model.prune_channels(layer, remaining_channels)
            new_model.to(device)

            optimizer = torch.optim.SGD([v for v in new_model.parameters() if v.requires_grad],
                                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
                                        
            finetune(new_model, optimizer, criterion, args.short_term_fine_tune, full_train_loader, layer, device)

            new_error = validate(val_loader, new_model, criterion, args)

            delta_acc = prev_acc - new_error # suppose accuracy will always drop. i.e. prev_acc@1 > now_acc@1
            cost_ratio = delta_acc / (new_gains/1000000) # accuracy losses / latency gains

            print(f"layer {layer} \t channels pruned {new_num_channels_pruned} \t error increase {delta_acc :.2f} \t "
                  f"predicted gains {new_gains/1000000 :.4f} \t ratio {cost_ratio :.2f}\n")

            prev_ratio = ((best_acc - prev_acc) / best_gains) if best_acc is not None else 0

            if best_acc is None or cost_ratio < prev_ratio: # if the pruned model now is more efficient than previous pruned model, set the model now as the best model.
                new_model.cpu()
                best_network, best_acc, pruned_layer = new_model, new_error, layer
                best_gains, number_pruned = new_gains, new_num_channels_pruned
            else:
                del new_model

        if best_network is None:
            raise Exception('We could not find a single layer to prune')
        print(f"the best validation error achieved was of {best_acc} for layer {pruned_layer};"
              f" {number_pruned} channels were pruned; inference time gains of {best_gains/1000000 :.4f}s")
        torch.cuda.empty_cache()  # frees GPU memory to avoid running out of ram
        best_network.to(device)
        model = best_network
        
        ### checkpoint
        torch.save(model, f'pruned_model{i}.pth')
        if args.record == 'True':
            ### record pruning process ###
            # Find total parameters and trainable parameters
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            data['pruned_layer_name'].append(pruned_layer[5:])
            data['acc1'].append(best_acc.cpu().numpy())
            data['target_gains'].append(f'{target_gains/1000000 :.4f}')
            data['gains'].append(f'{best_gains/1000000 :.4f}')
            data['num_of_pruned_channels'].append(number_pruned)
            data['rem_total_param'].append(total_params)
            data['rem_trainable_param'].append(total_trainable_params)
            df=DataFrame.from_dict(data, orient='index')
            df.to_excel('prune_result.xlsx')
            ##############################
        
        prev_acc = best_acc
        # prepare next step
        step_number += 1
        target_gains *= args.decay_rate
        
    # while loop end. Finish pruning.
    print(f"pruned network inference time according to perf_table: {model.total_cost/1000000 :.4f}s")
    # long term fine tune
    if args.long_term_fine_tune != 0:
        optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        finetune(model, optimizer, criterion, args.long_term_fine_tune, full_train_loader, "", device)
    print("finish pruning and long-term-fine-tune")
    # Save
    torch.save(model, 'pruned_model_final.pth')
    
    if args.record == 'True':
        ### record long_term_fine_tune result
        data['acc1'].append(validate(val_loader, model, criterion, args).cpu().numpy())
        df=DataFrame.from_dict(data, orient='index')
        df.to_excel('prune_result.xlsx')

