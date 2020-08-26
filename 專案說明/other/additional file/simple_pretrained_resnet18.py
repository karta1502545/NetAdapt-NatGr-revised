import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

batchSize = 100
resnet18 = models.resnet18(pretrained=True)
print(resnet18)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18.cuda()
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.Resize(96),
								transforms.ToTensor(),
								normalize])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
#testset = torchvision.datasets.ImageNet(root='./data', train=False, download=False, transform=transform)
print(len(testset)) #10000 testing photos
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=0)

#test 
resnet18.eval()
correct=0
total=0
for data in testloader:
    images,labels=data
    #print(images, labels)
    images=images.cuda()
    labels=labels.cuda()
    outputs=resnet18(Variable(images))
    _,predicted=torch.max(outputs,1)
    total+=labels.size(0)
    correct+=(predicted==labels).sum()
print('Accuracy of the network on the %d test images: %f %%' % (total , 100 * float(correct) / float(total)))
