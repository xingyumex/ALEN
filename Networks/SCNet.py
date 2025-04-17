import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class ScNet(nn.Module):

    def __init__(self):
        super(ScNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1,        out_channels=32,   kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels= 32,       out_channels=64,   kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels= 64,       out_channels=128,  kernel_size=3, padding=1)


        self.conv4 = nn.Conv2d(in_channels= 128, out_channels=256, kernel_size=3, padding=1)

        
        self.conv5 = nn.Conv2d(in_channels= 256, out_channels=128, kernel_size=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels= 256, out_channels=64,  kernel_size=1, padding=0)
        self.conv7 = nn.Conv2d(in_channels= 128, out_channels=32,   kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(in_channels= 64, out_channels=1,    kernel_size=1, padding=0)


    def forward(self, x):              
        x1 = F.relu(self.conv1(x))    
        x2 = F.relu(self.conv2(x1))    
        x3 = F.relu(self.conv3(x2))    

        x4 = F.relu(self.conv4(x3))  
        x5 = F.relu(self.conv5(x4))    
        x6 = torch.cat([x3,x5], dim=1) 

        x7 = F.relu(self.conv6(x6))  
        x8 = torch.cat([x2,x7], dim=1) 

        x9 = F.relu(self.conv7(x8))    
        x10 = torch.cat([x1,x9], dim=1) 

        x11 = F.relu(self.conv8(x10))  

        return x11


