import torch
import torch.nn as nn
import torch.nn.functional as F

class tbxrayCNN(nn.Module):
    def __init__(self):
        super().__init__()

        #Defining conv layers in Network
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        #Defining pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Defining the Full Connected Layer
        self.fc = nn.Linear(in_features=25088, out_features=2)

    def forward(self, x):
            
            #Layer 1
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool(x)

            #Layer 2
            x = self.conv2(x)
            x = F.relu(x)
            x = self.pool(x)

            #Layer 3
            x = self.conv3(x)
            x = F.relu(x)
            x = self.pool(x)

            #Flattening
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


            

