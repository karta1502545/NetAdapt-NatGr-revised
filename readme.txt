This is a project I am working on in ITRI in summer vacation.

The objective of this project is to combine NetAdapt thesis and some prominent model other than the original one(Wide Residual Net-40-2).

However, since the original code is hard to reuse when switching to the other networks, and I only have two months to finish a prototype, I choose to combine Residual Net-18 and NetAdapt algorithm together.

The code can process now; however, the accuracy is quite poor. That is, we can only learn how to prune a network using NetAdapt algorithm on ResNet-18. The result of running the code, which is a pruned model, is not a practical model to use, is the accuracy is quite low.

Thanks ITRI co-workers and the author of NetAdapt implementation.


Note:
By far only conv_x_y_1 can be pruned(see in resnet18.py "self.to_prune"). (x=2~5, y=1~2)
In addition, a layer cannot be pruned to 0 channels since the network structure limitation.
We can modify the minimum amount (>0) of remaining channels.(see in )
