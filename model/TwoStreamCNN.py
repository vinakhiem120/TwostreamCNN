import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision 
from torchvision.models import resnet50,resnet18,resnet152, ResNet50_Weights
    
class Convolution(nn.Module):
    def __init__(self,input = 1, output = 64):
        super().__init__()
        self.conv = nn.Conv2d(input,output, kernel_size=3)
        self.batch_norm = nn.BatchNorm2d(output)
        self.relu = nn.LeakyReLU()
    def forward(self,x):
        return self.relu(self.batch_norm(self.conv(x)))
    
class TwoStreamCNN(nn.Module):
    def __init__(self,num_classes,type='tsma'):
        super().__init__()
        self.conv1 = Convolution()
        self.conv2 = Convolution()
        self.conv3 = Convolution(128,3)
        self.type = type
        self.num_classes= num_classes
        self.relu = nn.LeakyReLU()
        #self.blockend = Resnet.ResNet50(num_classes= 29, channels= 128)
        self.block()
    def block(self):
        
        self.model = resnet152(pretrained = True)
            
        for param in self.model.parameters():
            param.requires_grad = True  
            
        fc_inputs = self.model.fc.in_features
        self.model.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fc_inputs,self.num_classes), # Since 29 possible outputs
            nn.Softmax(dim=1) 
        )
    def forward(self, streamA, streamB):
        ht = self.conv1(streamA)
        ht1 = self.conv2(streamB)
        z = torch.add(ht, ht1)

        if self.type == 'tsma':
            y = torch.cat((z, ht), dim=1)
        elif self.type == 'tsmb':
            y = torch.cat((z, ht1), dim=1)
        yhat = self.model(y)
        return yhat