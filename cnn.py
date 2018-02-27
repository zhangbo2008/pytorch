"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
torchvision
matplotlib
"""
# library
# standard library

import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 20
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False

a=os.listdir('./mnist')
print (a)
# Mnist digits dataset


train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=True,
)

# plot one example

'''这个是batch技术'''
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# convert test data into Variable, pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
#如果只有前向计算，没有后向梯度计算，设置volatile可以提高效率。所以对于test数据我们都要加volatile=True


test_y = test_data.test_labels[:2000]

'''建立网络的套路,这里面用class好麻烦,还不如直接都用Sequential来写'''
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        output = self.out(x)
        return output, x    # return x for visualization
'''这个view就是reshape的作用'''
# flatten the output of conv2 to (batch_size, 32 * 7 * 7)

cnn = CNN()
  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

'''下面是训练代码的套路,3引号这种批注不能在for循环里面用,很尴尬,很奇特'''
# training and testing
'''用enumerate来展开batch'''
for epoch in range(EPOCH):#控制整个数据循环几次

    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)
        # batch x
        b_y = Variable(y)   # batch y    cnn(b_x)[0]:20*10 他返回的是cnn里面forward函数里面的output 
        output = cnn(b_x)[0]              #cnn(b_x)[1]:20*1568他返回的是x  x是20*1568的因为32*7*7
        # cnn output
        
        loss = loss_func(output, b_y)# cross entropy loss
        
        #下面3行就是梯度反向传播,都封装好了
        optimizer.zero_grad()           # clear gradients for this training step  先让参数都归0
        loss.backward()                 # backpropagation, compute gradients    计算梯度
        optimizer.step()                # apply gradients   更新参数
        
        if step % 50 == 0: #每到50输出在test上结果
            test_output, last_layer = cnn(test_x)
#torch.max(x, n)[0] 沿着n维进行某种操作。得到的是某一维度的最大值之类的，如果不加维度n，则返回所有元素的最大值之类的
#n:0就是按照列找最大值 1就是按照行找最大值,这个跟网上博客写的不同可能是因为我版本高,修改了,torch.max(x, n)[1] 表示
#                                                                                         最大值所在的index.
#这里面用max然后取[1],直接找到每一行最大的概率是第几个index :torch.max(test_output, 1)[1]
            #teset_output:2000*10
            print (torch.max(test_output, 1)[1].data.squeeze())
            #squeeze把张量压缩成1维,为了算accuracy
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = sum(pred_y == test_y) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
#到此就结束了,下面的只是给出几个预测的实例很简单.
#总结一下:发现比tensorflow好写,思路清晰,速度也快,也是不错的框架,据说一些网络神马的没tensorflow好.对于我还是更实用的.
# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
