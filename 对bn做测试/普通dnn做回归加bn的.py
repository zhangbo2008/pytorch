"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
"""
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(nn.Module):
    def __init__(self):
            super(Net, self).__init__()
              #对初始数据归一化
            self.dnn = nn.Sequential(
            nn.BatchNorm1d(1, momentum=0.5),
            # input shape (1, 28, 28)
            nn.Linear(1, 100),                 # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                                         # output shape (16, 28, 28)
            #batchnormld的参数是上一层的输出
            nn.BatchNorm1d(100, momentum=0.5),# activation
            nn.ReLU(),
            nn.Linear(100, 10),                 # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                                         # output shape (16, 28, 28)
            
            nn.BatchNorm1d(10, momentum=0.5),# activation
            nn.ReLU(), # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Linear(10, 1), 


        )
    def forward(self, x):
            x=self.dnn(x)
            return x

net = Net()     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

plt.ion()   # something about plotting

for t in range(100):
    prediction = net(x)     # input x and predict based on x

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()



#下面我写一下输出的效果.代码几乎都相同只差一个bn添加没添加都只训练100步,看loss大小
#第一次:不加bn:0.0628  加:0.0038
## 第二次:      0.0568   加:0.0038
