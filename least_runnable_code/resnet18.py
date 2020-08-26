# This file defines all the pruning functions that main.py will use.
# and the method of building a (pretrained) ResNet-18 model
import torch
import torch.nn as nn
import torch.nn.functional as fct
from torch.hub import load_state_dict_from_url
import math
import numpy as np
import pickle

# get pretrained parameters
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    # copy all the prev_model in, out channels to rebuild the new model if prev_model is not None
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, prev_model=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)	

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

class ResNet(nn.Module):
    #block = Basicblock, layers=[2,2,2,2]
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.compute_metric_on = ["conv_1"] + [f"conv_{subnet_id}_{i}_{j}" for subnet_id in range(2, 6)
                                               for i in range(1, 3)
                                               for j in range(1, 3)] 
        self.all_layers = self.compute_metric_on +["fc"] + [f"downsample{i}" for i in range(2, 5)]

        # init model parameters
        self.n_channels = [64, 128, 256, 512]  # channel sizes
        self.drop_rate = 0.0
        self.perf_table = None
        self.num_classes = 1000
        self.total_cost = 0 # us->s
        self.to_prune = (['conv_2_1_1'] + ['conv_2_2_1']
                   + ['conv_3_1_1'] + ['conv_3_2_1']
                   + ['conv_4_1_1'] + ['conv_4_2_1']
                   + ['conv_5_1_1'] + ['conv_5_2_1']
			       )
		
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer_id=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], layer_id=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], layer_id=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], layer_id=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
	
    #_make_layer = make_subnetwork
    # block=BasicBlock(in "make_subnetwork" for loop), 
	# planes=self.n_channels[x], blocks=n_blocks_per_subnet,
	# stride=stride, dilate=???, layer_id=1~4
    def _make_layer(self, block, planes, blocks, layer_id, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # small block 1
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # inplanes = [64, 64, 128, 256]
        self.inplanes = planes * block.expansion
        
        # small block 2~N
        for _ in range(1, blocks): # blocks = 2 if resnet18
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def choose_num_channels(self, layer_name, cost_red_obj, allow_small_prunings):
        """chooses how many channels to remove
        :param layer_name: layer at which we remove channels
        :param cost_red_obj: the given cost reduction objective (to attain or approach as much as possible)
        :param allow_small_prunings: allows to prune layers for which we cannot achieve the reduction objective
        :returns: the number of channels to prune and achieved cost reduction or None, None if (we cannot achieve
        cost_red_obj and allow_small_prunings is set to False) or the Layer was pruned away"""
        layer = self.layer1[0].conv2
        if layer_name == 'conv_2_1_1':
            layer = self.layer1[0].conv1
        elif layer_name == 'conv_2_1_2':
            layer = self.layer1[0].conv2
        elif layer_name == 'conv_2_2_1':
            layer = self.layer1[1].conv1
        elif layer_name == 'conv_2_2_2':
            layer = self.layer1[1].conv2
        elif layer_name == 'conv_3_1_1':
            layer = self.layer2[0].conv1
        elif layer_name == 'conv_3_1_2':
            layer = self.layer2[0].conv2
        elif layer_name == 'conv_3_2_1':
            layer = self.layer2[1].conv1
        elif layer_name == 'conv_3_2_2':
            layer = self.layer2[1].conv2
        elif layer_name == 'conv_4_1_1':
            layer = self.layer3[0].conv1
        elif layer_name == 'conv_4_1_2':
            layer = self.layer3[0].conv2
        elif layer_name == 'conv_4_2_1':
            layer = self.layer3[1].conv1
        elif layer_name == 'conv_4_2_2':
            layer = self.layer3[1].conv2
        elif layer_name == 'conv_5_1_1':
            layer = self.layer4[0].conv1
        elif layer_name == 'conv_5_1_2':
            layer = self.layer4[0].conv2
        elif layer_name == 'conv_5_2_1':
            layer = self.layer4[1].conv1
        elif layer_name == 'conv_5_2_2':
            layer = self.layer4[1].conv2
        
        if layer is None:
            print(f"{layer_name} has been pruned away")
            return None, None

        init_nbr_out_channels = layer.out_channels
        print(f"{layer_name} has {init_nbr_out_channels} output channels")
        #print(f'start get_cost_array layername={layer_name}')
        costs_array = self._get_cost_array(layer_name)
        #print('end get_cost_array')
        # determines the number of filters
        prev_cost = costs_array[init_nbr_out_channels - 1]
        minimum_channel_per_layer = 10
        for rem_channels in range(init_nbr_out_channels - 1, minimum_channel_per_layer-1, -1):  # could not be made O(log(n)) instead because cost
            # table not strictly monotonic
            cost_diff = prev_cost - costs_array[rem_channels - 1]  # -1 because we are directly accessing a cost_array
            if cost_diff > cost_red_obj:
                return init_nbr_out_channels - rem_channels, cost_diff
        print(f"{layer_name} cannot be pruned anymore")
        return None, None

    def load_table(self, name):
        """loads the pickle file coresponding to table and prints the total cost of the network"""
        with open(name, 'rb') as file:
            self.perf_table = pickle.load(file)

        self._compute_total_cost()

        print(f"the total cost of the model is: {self.total_cost/1000000 :.3f}s according to the perf table")

    def _get_cost_array(self, layer_name):
        """returns the costs array corresponding to the layer named laryer_name, the difference between cost_array[a]
        and cost_array[b] is the performance gain according to the perf_tables when pruning from a to b channels in the
        layer named laryer_name"""
        '''now prune conv_x_y_1 only.'''
        #else:  # layer_name = Conv_x_y_1, we only influence Conv_x_y_2
        # since we got here, Conv_x_y_1 cannot be none
        layer_name_table = f"No_Stride_{int(layer_name[5])-1}"
        next_layer = layer_name[:9] + "2"
        subnet_layer_name = f"layer{int(layer_name[5])-1}"
        layer1_in_channels = getattr(self, subnet_layer_name)[int(layer_name[7])-1].conv1.in_channels
        layer2_out_channels = getattr(self, subnet_layer_name)[int(layer_name[7])-1].conv2.out_channels
        
        
        costs_array = self.get_cost(layer_name_table, layer1_in_channels, None) + \
            self.get_cost(layer_name_table, None, layer2_out_channels)

        return costs_array
	
    def get_cost(self, layer_name, in_channel, out_channel):
        """ get the cost of layer layer_name when it has in_channel and out_channel channels, None means we return all
        input channels"""
        if in_channel is None:
            if out_channel is None:
                return self.perf_table[layer_name][:, :]
            else:
                return self.perf_table[layer_name][:, out_channel - 1]
        elif out_channel is None:
            return self.perf_table[layer_name][in_channel - 1, :]
        else:
            return self.perf_table[layer_name][in_channel - 1, out_channel - 1]

    def _compute_total_cost(self):
        """computes the total cost of the network by adding the costs of the channels in the perf_table"""
        self.total_cost = 0
        for layer_name in self.all_layers: # TODO
            if layer_name == "fc":
                self.total_cost += self.get_cost("FC", self.fc.in_features, 1)
            elif layer_name[:10] == "downsample": #self.layerx[0].downsample[0].in_channels
                subnet_layer_name = "layer" + layer_name[10]
                din = getattr(self, subnet_layer_name)[0].downsample[0].in_channels
                dout = getattr(self, subnet_layer_name)[0].downsample[0].out_channels
                self.total_cost += self.get_cost(f"DownSampling_{layer_name[10]}", din, dout)
            elif layer_name == 'conv_1':
                self.total_cost += self.get_cost("Conv_1", self.conv1.in_channels, self.conv1.out_channels)
            else:
                layer = self.layer1[0].conv2
                if layer_name == 'conv_2_1_1':
                    layer = self.layer1[0].conv1
                elif layer_name == 'conv_2_1_2':
                    layer = self.layer1[0].conv2
                elif layer_name == 'conv_2_2_1':
                    layer = self.layer1[1].conv1
                elif layer_name == 'conv_2_2_2':
                    layer = self.layer1[1].conv2
                elif layer_name == 'conv_3_1_1':
                    layer = self.layer2[0].conv1
                elif layer_name == 'conv_3_1_2':
                    layer = self.layer2[0].conv2
                elif layer_name == 'conv_3_2_1':
                    layer = self.layer2[1].conv1
                elif layer_name == 'conv_3_2_2':
                    layer = self.layer2[1].conv2
                elif layer_name == 'conv_4_1_1':
                    layer = self.layer3[0].conv1
                elif layer_name == 'conv_4_1_2':
                    layer = self.layer3[0].conv2
                elif layer_name == 'conv_4_2_1':
                    layer = self.layer3[1].conv1
                elif layer_name == 'conv_4_2_2':
                    layer = self.layer3[1].conv2
                elif layer_name == 'conv_5_1_1':
                    layer = self.layer4[0].conv1
                elif layer_name == 'conv_5_1_2':
                    layer = self.layer4[0].conv2
                elif layer_name == 'conv_5_2_1':
                    layer = self.layer4[1].conv1
                elif layer_name == 'conv_5_2_2':
                    layer = self.layer4[1].conv2
                else:
                    layer = None

                if layer is not None:
                    if layer_name.endswith("_1_1"):#conv_5_1_1 = self.layer4[0].conv1
                        layer_name_table = f"Stride_{int(layer_name[5])-1}"
                    else:
                        layer_name_table = f"No_Stride_{int(layer_name[5])-1}"

                    self.total_cost += self.get_cost(layer_name_table, layer.in_channels, layer.out_channels)

    def remove_from_in_channels(self, layer_name, remaining_channels):
        """removes the input channels to the layer
        :param layer_name: the name of the layer that will be pruned
        :param remaining_channels: the offsets of the remaining channels"""
        # get attribute
        #print(f'Hello from rfinchannels! layer_name = {layer_name}')
        layer = self.layer1[0].conv2
        if layer_name == 'conv_2_1_1':
            layer = self.layer1[0].conv1
        elif layer_name == 'conv_2_1_2':
            layer = self.layer1[0].conv2
        elif layer_name == 'conv_2_2_1':
            layer = self.layer1[1].conv1
        elif layer_name == 'conv_2_2_2':
            layer = self.layer1[1].conv2
        elif layer_name == 'conv_3_1_1':
            layer = self.layer2[0].conv1
        elif layer_name == 'conv_3_1_2':
            layer = self.layer2[0].conv2
        elif layer_name == 'conv_3_2_1':
            layer = self.layer2[1].conv1
        elif layer_name == 'conv_3_2_2':
            layer = self.layer2[1].conv2
        elif layer_name == 'conv_4_1_1':
            layer = self.layer3[0].conv1
        elif layer_name == 'conv_4_1_2':
            layer = self.layer3[0].conv2
        elif layer_name == 'conv_4_2_1':
            layer = self.layer3[1].conv1
        elif layer_name == 'conv_4_2_2':
            layer = self.layer3[1].conv2
        elif layer_name == 'conv_5_1_1':
            layer = self.layer4[0].conv1
        elif layer_name == 'conv_5_1_2':
            layer = self.layer4[0].conv2
        elif layer_name == 'conv_5_2_1':
            layer = self.layer4[1].conv1
        elif layer_name == 'conv_5_2_2':
            layer = self.layer4[1].conv2
        else:
            raise Exception('Error from "remove_from_in_channels"')
        num_remaining_channels = remaining_channels.size(0)
        if layer_name == "FC":
            new_layer = nn.Linear(num_remaining_channels, self.num_classes)
            new_layer.weight.data = layer.weight.data[:, remaining_channels]
            new_layer.bias.data = layer.bias.data
        else:
            new_layer = nn.Conv2d(in_channels=num_remaining_channels, out_channels=layer.out_channels,
                                  kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                                  dilation=layer.dilation, groups=layer.groups, bias=layer.bias is not None)
            new_layer.weight.data = layer.weight.data[:, remaining_channels, :, :]  # out_ch * in_ch * height * width
            if new_layer.bias is not None:
                new_layer.bias.data = layer.bias.data
        # set attribute
        if layer_name == 'conv_2_1_1':
            self.layer1[0].conv1 = new_layer
        elif layer_name == 'conv_2_1_2':
            self.layer1[0].conv2 = new_layer
        elif layer_name == 'conv_2_2_1':
            self.layer1[1].conv1 = new_layer
        elif layer_name == 'conv_2_2_2':
            self.layer1[1].conv2 = new_layer
        elif layer_name == 'conv_3_1_1':
            self.layer2[0].conv1 = new_layer
        elif layer_name == 'conv_3_1_2':
            self.layer2[0].conv2 = new_layer
        elif layer_name == 'conv_3_2_1':
            self.layer2[1].conv1 = new_layer
        elif layer_name == 'conv_3_2_2':
            self.layer2[1].conv2 = new_layer
        elif layer_name == 'conv_4_1_1':
            self.layer3[0].conv1 = new_layer
        elif layer_name == 'conv_4_1_2':
            self.layer3[0].conv2 = new_layer
        elif layer_name == 'conv_4_2_1':
            self.layer3[1].conv1 = new_layer
        elif layer_name == 'conv_4_2_2':
            self.layer3[1].conv2 = new_layer
        elif layer_name == 'conv_5_1_1':
            self.layer4[0].conv1 = new_layer
        elif layer_name == 'conv_5_1_2':
            self.layer4[0].conv2 = new_layer
        elif layer_name == 'conv_5_2_1':
            self.layer4[1].conv1 = new_layer
        elif layer_name == 'conv_5_2_2':
            self.layer4[1].conv2 = new_layer
        
    def remove_from_out_channels(self, layer_name, remaining_channels): 
        """removes the output channels to the layer and adapts the corresponding batchnorm layers (in the case of
        Skip_x, Conv_x_0_2_bn is left unchanged)
        :param layer_name: the name of the layer that will be pruned
        :param remaining_channels: the offsets of the remaining channels"""
        layer =  self.layer1[0].conv1 # conv_2_1_1
        if layer_name == 'conv_2_1_1':
            layer = self.layer1[0].conv1
        elif layer_name == 'conv_2_1_2':
            layer = self.layer1[0].conv2
        elif layer_name == 'conv_2_2_1':
            layer = self.layer1[1].conv1
        elif layer_name == 'conv_2_2_2':
            layer = self.layer1[1].conv2
        elif layer_name == 'conv_3_1_1':
            layer = self.layer2[0].conv1
        elif layer_name == 'conv_3_1_2':
            layer = self.layer2[0].conv2
        elif layer_name == 'conv_3_2_1':
            layer = self.layer2[1].conv1
        elif layer_name == 'conv_3_2_2':
            layer = self.layer2[1].conv2
        elif layer_name == 'conv_4_1_1':
            layer = self.layer3[0].conv1
        elif layer_name == 'conv_4_1_2':
            layer = self.layer3[0].conv2
        elif layer_name == 'conv_4_2_1':
            layer = self.layer3[1].conv1
        elif layer_name == 'conv_4_2_2':
            layer = self.layer3[1].conv2
        elif layer_name == 'conv_5_1_1':
            layer = self.layer4[0].conv1
        elif layer_name == 'conv_5_1_2':
            layer = self.layer4[0].conv2
        elif layer_name == 'conv_5_2_1':
            layer = self.layer4[1].conv1
        elif layer_name == 'conv_5_2_2':
            layer = self.layer4[1].conv2
        else:
            layer = None
		
        num_remaining_channels = remaining_channels.size(0)

        # conv layer
        new_layer = nn.Conv2d(in_channels=layer.in_channels, out_channels=num_remaining_channels,
                              kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups, bias=layer.bias is not None)

        new_layer.weight.data = layer.weight.data[remaining_channels, :, :, :]  # out_ch * in_ch * height * width
        if new_layer.bias is not None:
            new_layer.bias.data = layer.bias.data[remaining_channels]
        
        # set attribute
        if layer_name == 'conv_2_1_1':
            self.layer1[0].conv1 = new_layer
        elif layer_name == 'conv_2_1_2':
            self.layer1[0].conv2 = new_layer
        elif layer_name == 'conv_2_2_1':
            self.layer1[1].conv1 = new_layer
        elif layer_name == 'conv_2_2_2':
            self.layer1[1].conv2 = new_layer
        elif layer_name == 'conv_3_1_1':
            self.layer2[0].conv1 = new_layer
        elif layer_name == 'conv_3_1_2':
            self.layer2[0].conv2 = new_layer
        elif layer_name == 'conv_3_2_1':
            self.layer2[1].conv1 = new_layer
        elif layer_name == 'conv_3_2_2':
            self.layer2[1].conv2 = new_layer
        elif layer_name == 'conv_4_1_1':
            self.layer3[0].conv1 = new_layer
        elif layer_name == 'conv_4_1_2':
            self.layer3[0].conv2 = new_layer
        elif layer_name == 'conv_4_2_1':
            self.layer3[1].conv1 = new_layer
        elif layer_name == 'conv_4_2_2':
            self.layer3[1].conv2 = new_layer
        elif layer_name == 'conv_5_1_1':
            self.layer4[0].conv1 = new_layer
        elif layer_name == 'conv_5_1_2':
            self.layer4[0].conv2 = new_layer
        elif layer_name == 'conv_5_2_1':
            self.layer4[1].conv1 = new_layer
        elif layer_name == 'conv_5_2_2':
            self.layer4[1].conv2 = new_layer

        # batchnorm layer # If I prune some channels of self.layer1[0].conv1(conv2_1_1), self.layer1[0].bn1(conv2_1_1_bn) should be modified at the same time.
        bn_layer = self.layer1[0].bn1 # initialize bn_layer
        if layer_name[:4] != "Skip":
            # get attribute of batchnorm layer (bn)
            if layer_name == 'conv_2_1_1':
                bn_layer = self.layer1[0].bn1
            elif layer_name == 'conv_2_1_2':
                bn_layer = self.layer1[0].bn2
            elif layer_name == 'conv_2_2_1':
                bn_layer = self.layer1[1].bn1
            elif layer_name == 'conv_2_2_2':
                bn_layer = self.layer1[1].bn2
            elif layer_name == 'conv_3_1_1':
                bn_layer = self.layer2[0].bn1
            elif layer_name == 'conv_3_1_2':
                bn_layer = self.layer2[0].bn2
            elif layer_name == 'conv_3_2_1':
                bn_layer = self.layer2[1].bn1
            elif layer_name == 'conv_3_2_2':
                bn_layer = self.layer2[1].bn2
            elif layer_name == 'conv_4_1_1':
                bn_layer = self.layer3[0].bn1
            elif layer_name == 'conv_4_1_2':
                bn_layer = self.layer3[0].bn2
            elif layer_name == 'conv_4_2_1':
                bn_layer = self.layer3[1].bn1
            elif layer_name == 'conv_4_2_2':
                bn_layer = self.layer3[1].bn2
            elif layer_name == 'conv_5_1_1':
                bn_layer = self.layer4[0].bn1
            elif layer_name == 'conv_5_1_2':
                bn_layer = self.layer4[0].bn2
            elif layer_name == 'conv_5_2_1':
                bn_layer = self.layer4[1].bn1
            elif layer_name == 'conv_5_2_2':
                bn_layer = self.layer4[1].bn2
            
            new_bn_layer = nn.BatchNorm2d(num_remaining_channels, eps=bn_layer.eps, momentum=bn_layer.momentum,
                                          affine=bn_layer.affine, track_running_stats=bn_layer.track_running_stats)
            if new_bn_layer.affine:
                new_bn_layer.weight.data = bn_layer.weight.data[remaining_channels]
                new_bn_layer.bias.data = bn_layer.bias.data[remaining_channels]

            if new_bn_layer.track_running_stats:
                new_bn_layer.running_mean.data = bn_layer.running_mean.data[remaining_channels]
                new_bn_layer.running_var.data = bn_layer.running_var.data[remaining_channels]
                new_bn_layer.num_batches_tracked.data = bn_layer.num_batches_tracked.data
            # set attribute of batchnorm layer (bn)
            if layer_name == 'conv_2_1_1':
                self.layer1[0].bn1 = new_bn_layer
            elif layer_name == 'conv_2_1_2':
                self.layer1[0].bn2 = new_bn_layer
            elif layer_name == 'conv_2_2_1':
                self.layer1[1].bn1 = new_bn_layer
            elif layer_name == 'conv_2_2_2':
                self.layer1[1].bn2 = new_bn_layer
            elif layer_name == 'conv_3_1_1':
                self.layer2[0].bn1 = new_bn_layer
            elif layer_name == 'conv_3_1_2':
                self.layer2[0].bn2 = new_bn_layer
            elif layer_name == 'conv_3_2_1':
                self.layer2[1].bn1 = new_bn_layer
            elif layer_name == 'conv_3_2_2':
                self.layer2[1].bn2 = new_bn_layer
            elif layer_name == 'conv_4_1_1':
                self.layer3[0].bn1 = new_bn_layer
            elif layer_name == 'conv_4_1_2':
                self.layer3[0].bn2 = new_bn_layer
            elif layer_name == 'conv_4_2_1':
                self.layer3[1].bn1 = new_bn_layer
            elif layer_name == 'conv_4_2_2':
                self.layer3[1].bn2 = new_bn_layer
            elif layer_name == 'conv_5_1_1':
                self.layer4[0].bn1 = new_bn_layer
            elif layer_name == 'conv_5_1_2':
                self.layer4[0].bn2 = new_bn_layer
            elif layer_name == 'conv_5_2_1':
                self.layer4[1].bn1 = new_bn_layer
            elif layer_name == 'conv_5_2_2':
                self.layer4[1].bn2 = new_bn_layer
            #else:
                # error
    def choose_which_channels(self, layer_name, num_channels):
        """chooses which channels remains after the pruning
        :param layer_name: layer at which we remove filters
        :param num_channels: the number of channels to remove
        :returns: the id of the channels that survives the pruning"""
        pruning_score = self.get_pruning_score(layer_name)
        #print(f'From choose_which_channels: {self.layer1[0].conv1}')

        # get the channels with biggest score
        _, remaining_channels = torch.topk(pruning_score, k=pruning_score.size(0) - num_channels, largest=True, dim=0)

        remaining_channels, _ = torch.sort(remaining_channels)

        return remaining_channels

    def get_pruning_score(self, layer_name):
        """get the pruning score (the bigger the less interesting to prune) associated with each channel of the layer
        :param layer_name: name of the concerned layer
        :returns: the pruning_score associated with each channel"""
        # initialize weights as weights of conv_2_1_1
        #weights = getattr(self, layer_name).weight.data  # out_ch * in_ch * height * width
        weights = self.layer1[0].conv1.weight.data # conv_2_1_1
        if layer_name == 'conv_2_1_1':
            weights = self.layer1[0].conv1.weight.data
        elif layer_name == 'conv_2_1_2':
            weights = self.layer1[0].conv2.weight.data
        elif layer_name == 'conv_2_2_1':
            weights = self.layer1[1].conv1.weight.data
        elif layer_name == 'conv_2_2_2':
            weights = self.layer1[1].conv2.weight.data
        elif layer_name == 'conv_3_1_1':
            weights = self.layer2[0].conv1.weight.data
        elif layer_name == 'conv_3_1_2':
            weights = self.layer2[0].conv2.weight.data
        elif layer_name == 'conv_3_2_1':
            weights = self.layer2[1].conv1.weight.data
        elif layer_name == 'conv_3_2_2':
            weights = self.layer2[1].conv2.weight.data
        elif layer_name == 'conv_4_1_1':
            weights = self.layer3[0].conv1.weight.data
        elif layer_name == 'conv_4_1_2':
            weights = self.layer3[0].conv2.weight.data
        elif layer_name == 'conv_4_2_1':
            weights = self.layer3[1].conv1.weight.data
        elif layer_name == 'conv_4_2_2':
            weights = self.layer3[1].conv2.weight.data
        elif layer_name == 'conv_5_1_1':
            weights = self.layer4[0].conv1.weight.data
        elif layer_name == 'conv_5_1_2':
            weights = self.layer4[0].conv2.weight.data
        elif layer_name == 'conv_5_2_1':
            weights = self.layer4[1].conv1.weight.data
        elif layer_name == 'conv_5_2_2':
            weights = self.layer4[1].conv2.weight.data
        #else:
            # error
    
        weights_norm = torch.norm(weights.view(weights.size()[0], -1), p=2, dim=1)
        num_layers = 1
        pruning_score = weights_norm.div(num_layers)

        return pruning_score

    def prune_channels(self, layer_name, remaining_channels): # TODO: try to implement pruning every layer
        """modifies the structure of self so as to remove the given channels at the given layer
        :param layer_name: the name of the layer that will be pruned
        :param remaining_channels: the offsets of the remaining channels"""
        # name == conv_x_y_1, we only influence conv_x_y_2
        next_layer_name = layer_name[:9] + "2"
        if remaining_channels.size(0) != 0:
            self.remove_from_out_channels(layer_name, remaining_channels)
            self.remove_from_in_channels(next_layer_name, remaining_channels)
        else:
            # set attribute
            if layer_name == 'conv_2_1_1':
                self.layer1[0].conv1 = None
            elif layer_name == 'conv_2_1_2':
                self.layer1[0].conv2 = None
            elif layer_name == 'conv_2_2_1':
                self.layer1[1].conv1 = None
            elif layer_name == 'conv_2_2_2':
                self.layer1[1].conv2 = None
            elif layer_name == 'conv_3_1_1':
                self.layer2[0].conv1 = None
            elif layer_name == 'conv_3_1_2':
                self.layer2[0].conv2 = None
            elif layer_name == 'conv_3_2_1':
                self.layer2[1].conv1 = None
            elif layer_name == 'conv_3_2_2':
                self.layer2[1].conv2 = None
            elif layer_name == 'conv_4_1_1':
                self.layer3[0].conv1 = None
            elif layer_name == 'conv_4_1_2':
                self.layer3[0].conv2 = None
            elif layer_name == 'conv_4_2_1':
                self.layer3[1].conv1 = None
            elif layer_name == 'conv_4_2_2':
                self.layer3[1].conv2 = None
            elif layer_name == 'conv_5_1_1':
                self.layer4[0].conv1 = None
            elif layer_name == 'conv_5_1_2':
                self.layer4[0].conv2 = None
            elif layer_name == 'conv_5_2_1':
                self.layer4[1].conv1 = None
            elif layer_name == 'conv_5_2_2':
                self.layer4[1].conv2 = None

            if next_layer_name == 'conv_2_1_1':
                self.layer1[0].conv1 = None
            elif next_layer_name == 'conv_2_1_2':
                self.layer1[0].conv2 = None
            elif next_layer_name == 'conv_2_2_1':
                self.layer1[1].conv1 = None
            elif next_layer_name == 'conv_2_2_2':
                self.layer1[1].conv2 = None
            elif next_layer_name == 'conv_3_1_1':
                self.layer2[0].conv1 = None
            elif next_layer_name == 'conv_3_1_2':
                self.layer2[0].conv2 = None
            elif next_layer_name == 'conv_3_2_1':
                self.layer2[1].conv1 = None
            elif next_layer_name == 'conv_3_2_2':
                self.layer2[1].conv2 = None
            elif next_layer_name == 'conv_4_1_1':
                self.layer3[0].conv1 = None
            elif next_layer_name == 'conv_4_1_2':
                self.layer3[0].conv2 = None
            elif next_layer_name == 'conv_4_2_1':
                self.layer3[1].conv1 = None
            elif next_layer_name == 'conv_4_2_2':
                self.layer3[1].conv2 = None
            elif next_layer_name == 'conv_5_1_1':
                self.layer4[0].conv1 = None
            elif next_layer_name == 'conv_5_1_2':
                self.layer4[0].conv2 = None
            elif next_layer_name == 'conv_5_2_1':
                self.layer4[1].conv1 = None
            elif next_layer_name == 'conv_5_2_2':
                self.layer4[1].conv2 = None
