This is a project I am working on in ITRI in Taiwan during the 2020 summer vacation.

The objective of this project is to combine NetAdapt thesis and some prominent model other than the original one(Wide Residual Net-40-2).
The task is to reduce numbers of parameters in a model, which can speed up model prediction, while the accuracy doesn't decrease too much.
However, since the original code is hard to reuse when switching to the other networks, and I only have two months to finish a prototype, I choose to combine Residual Net-18 and NetAdapt algorithm together.

The code can process now. (pretrained:69.14% -> 67.2% while 30% of latency reduced)
The process of pruning on ResNet18 can be seen obviously.

It really take a lot effort to revise the code of NetAdapt Implementation written by NatGr.
Since the code is old when I try to revise it(2020), there may exist some other efficient way to prune a network. (some strong api may be useful in PyTorch library)

Thanks ITRI co-workers, the author of NetAdapt implementation, and all the people that have helped me out.

Note:
By far only conv_x_y_1 can be pruned(see in resnet18.py "self.to_prune"). (x=2~5, y=1~2)
In addition, a layer cannot be pruned to 0 channels since the network structure limitation.
We can modify the minimum amount (>0) of remaining channels(i.e. filters). (see in resnet18.py "minimum_channel_per_layer")

* More details are available in the powerpoint I provided above.
