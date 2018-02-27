from torch.autograd import Variable
import torch
x = Variable(torch.ones(2, 2), requires_grad = True)
y = x + 2


# y 是作为一个操作的结果创建的因此y有一个creator 
z = y * y * 3
out = z.mean()

# 现在我们来使用反向传播



print (out.backward(torch.Tensor([1.0])))


help(out.backward)

# out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
# 在此处输出 d(out)/dx
x.grad
print (x.grad)
#这自动求导确实强
x = torch.randn(3)
x = Variable(x, requires_grad = True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print (x.grad)
















import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()



print (net)
params = list(net.parameters())
help(nn.Conv2d)  #默认stride=1, padding=0,dilation=1, groups=1, bias=True)
print(len(params))  #10
print(params[0]) # conv1's .weight  #torch.Size([6, 1, 5, 5])
print(params[1]) # conv1's .weight  #torch.Size([6])第二层为什么是6?
                 #他表示的是啥?他应该表示的是bias,bias默认是True,因为上一层输出6个所以这里面的bias是6个标量组成的tensor
print(params[2]) # conv1's .weight  #torch.Size([16,1,5,5])
print(params[3]) # 16
print(params[4]) # [torch.FloatTensor of size 120x400]   size=out_put*in_put
print(params[5]) #bias 120
print(params[6]) #[torch.FloatTensor of size 84x120]
print(params[7]) #84
print(params[8]) #10x84]
print(params[9]) #10   ,总共就这么多层.



input = Variable(torch.randn(1, 1, 32, 32))   #第一个分量是batch_size,










input=Variable(torch.randn( 1, 32, 32))   #第二种方法是放入一个图片,1表示颜色channel,后面32*32表示大小
input=input.unsqueeze(0)                  #为了训练把他变成batch,用unsqueeze放大一个0维度即可
print (input)                             #变成了1*1*32*32


out = net(input)
'''out 的输出结果如下
Variable containing:
-0.0158 -0.0682 -0.1239 -0.0136 -0.0645  0.0107 -0.0230 -0.0085  0.1172 -0.0393
[torch.FloatTensor of size 1x10]
'''

net.zero_grad() # 对所有的参数的梯度缓冲区进行归零
out.backward(torch.randn(1, 10)) # 使用随机的梯度进行反向传播



##那么数据怎么办呢？
##
##通常来讲，当你处理图像，声音，文本，视频时需要使用python中其他独立的包来将他们转换为numpy中的数组，之后再转换为torch.*Tensor。
##
##图像的话，可以用Pillow, OpenCV。
##声音处理可以用scipy和librosa。
##文本的处理使用原生Python或者Cython以及NLTK和SpaCy都可以。
##
##
##特别的对于图像，我们有torchvision这个包可用,其中包含了一些现
##成的数据集如：Imagenet, CIFAR10, MNIST等等。同时还有一些转换图像用的工具。 这非常的方便并且避免了写样板代码。






























