import torch
from torchvision import transforms
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        
        #initialize first layer:conv->relu-->maxpooling
        self.conv1=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize second layer:conv->relu-->maxpooling
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize third layer:FC layer-->relu
        self.fc1=nn.Linear(in_features=16*5*5, out_features=120)
        self.relu3=nn.ReLU()
        
        #initialize fourth layer:FC layer-->relu
        self.fc2=nn.Linear(in_features=120, out_features=84)
        self.relu4=nn.ReLU()
        
        #initialize fifth layer:FC layer-->logSoftMax
        self.fc3=nn.Linear(in_features=84, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        #passing the input through first set of layers
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        #passing output from first layer through second set of layers
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        
        #passing output from second layer through third set of layers
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.relu3(x)
        
        #passing output from third layer through fourth set of layers
        x=self.fc2(x)
        x=self.relu4(x)
        
        #passing output from fourth layer through fifth set of layers
        x=self.fc3(x)
        output = self.logSoftmax(x)
        return output

class LeNet5_BN(nn.Module):
    def __init__(self):
        super().__init__()
        
        #initialize first layer:conv->BatchNormalization-->relu-->maxpooling
        self.conv1=nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.bn1=nn.BatchNorm2d(6)
        self.relu1=nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize second layer:conv->BatchNormalization-->relu-->maxpooling
        self.conv2=nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.bn2=nn.BatchNorm2d(16)
        self.relu2=nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #initialize third layer:FC layer-->BatchNormalization-->relu
        self.fc1=nn.Linear(in_features=16*5*5, out_features=120)
        self.bn3=nn.BatchNorm1d(120)
        self.relu3=nn.ReLU()
        
        #initialize fourth layer:FC layer-->BatchNormalization-->relu
        self.fc2=nn.Linear(in_features=120, out_features=84)
        self.bn4=nn.BatchNorm1d(84)
        self.relu4=nn.ReLU()
        
        #initialize fifth layer:FC layer-->logSoftMax
        self.fc3=nn.Linear(in_features=84, out_features=10)
        self.logSoftmax = nn.LogSoftmax(dim=1)
    
    def forward(self,x):
        #passing the input through first set of layers
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.pool1(x)
        
        #passing output from first layer through second set of layers
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        
        #passing output from second layer through third set of layers
        x=torch.flatten(x,1)
        x=self.fc1(x)
        x=self.bn3(x)
        x=self.relu3(x)
        
        #passing output from third layer through fourth set of layers
        x=self.fc2(x)
        x=self.bn4(x)
        x=self.relu4(x)
        
        #passing output from fourth layer through fifth set of layers
        x=self.fc3(x)
        output = self.logSoftmax(x)
        return output
