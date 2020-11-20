import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1   = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2   = nn.Linear(in_features=120,    out_features=60)
        self.out   = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        ''' Flatten before dense layer'''
        t = t.flatten(1, -1) #(start_dim, end_dim)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = self.out(t)
        #t = F.softmax(t)
        '''we'll use loss fxn = cross_entropy
        which has inbuild softmax calculation.
        [vs cross_entropy_with_logits]'''
        return t
    
net = MyNet()
net.forward(torch.rand(10,1,28,28))
print(net)
