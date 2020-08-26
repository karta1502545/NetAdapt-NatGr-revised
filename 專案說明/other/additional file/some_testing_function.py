'''
# initialize parameters in a model
for name, module in model.named_children():
    print('resetting ', name)
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
'''
''' #replaced by deepcopy
def build_model(prev_model=None):
    """build the model given the arguments on the device, take care that you still need to call model.to(device)
    if prev_model is not None, we will copy the number of channels in all the layers of prev_model"""
    return resnet18(prev_model)
'''
'''
# try to disappear a layer and load in parameters
model.layer3[0].conv1 = None
checkpoint = torch.load(os.path.join('checkpoints', 'ResNet18.pth'),
                                 map_location='cpu')['state_dict']
model.load_state_dict(checkpoint, strict=True)
'''

#model.load_state_dict(torch.load(os.path.join('checkpoints', f'{args.base_model}.pth'),
#                                 map_location='cpu')['state_dict'], strict=True)