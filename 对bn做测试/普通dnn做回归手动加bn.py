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
            
            self.diyi=nn.Linear(1, 100)
            self.dier=nn.Linear(100, 10) # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                                       # output shape (16, 28, 28)
            #batchnormld的参数是上一层的输出
            
            self.diyi1=nn.ReLU()
             # choose max value in 2x2 area, output shape (16, 14, 14)
            self.disan=nn.Linear(10, 1)


            self.ui=nn.Linear(100,100)
            self.ui2=nn.Linear(10,10)
    def forward(self, x):
            m = torch.mean(x, dim=0)#计算均值 注意是在batch_size这个dim上做mean.
            std = torch.std(x, dim=0)#计算标准差
            epsilon=0.0 #必须写的足够小才能归一化,我写0.01都不行,这个需要测试.
            x_normed = (x - m) / (std + epsilon)#归一化
            
            x=x_normed
            x=self.diyi(x)
            
            
            m = torch.mean(x, dim=0)#计算均值 注意是在batch_size这个dim上做mean.
            std = torch.std(x, dim=0)#计算标准差
            epsilon=0.0 #必须写的足够小才能归一化,我写0.01都不行,这个需要测试.
            x_normed = (x - m) / (std + epsilon)#归一化
            
            x=x_normed
            x=self.ui(x)
            x=self.diyi1(x)
            
            x=self.dier(x)
            

            m = torch.mean(x, dim=0)#计算均值 注意是在batch_size这个dim上做mean.
            std = torch.std(x, dim=0)#计算标准差
            epsilon=0.0 #必须写的足够小才能归一化,我写0.01都不行,这个需要测试.
            x_normed = (x - m) / (std + epsilon)#归一化
            
            x=x_normed
            x=self.ui2(x)
            x=self.diyi1(x)
            x=self.disan(x)
            
            
            
            
            return x

net = Net()     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
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




#下面我写一下输出的效果.代码几乎都相同只差一个bn添加没添加



#手动加没写明白,跑是跑了,效果这么差?
#后来经过我的改动,写出来了,原因就是SGD的理解不够,他里面第一个参数是
## net.parameters(),他表示他只训练网络里面的系数,这个系数指的是__init__里面的系数
## 不是forward里面的系数,所以forward里面的网络除了归一化以外的网络都需要在__init__
## 里面先生成这个网络模型,然后他的参数才能给SGD,然后才能进行学习,
#效果差不多,手动写的效果更好一点,可能是因为我的学习率改更小了,说明学习率还是很重要的
#即使你用了bn层也一定要调试好这个超参数
