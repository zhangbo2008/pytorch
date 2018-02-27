"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
numpy
"""
#莫烦的bn算法
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# torch.manual_seed(1)    # reproducible
# np.random.seed(1)

# Hyper parameters
N_SAMPLES = 2000
BATCH_SIZE = 20
EPOCH = 12
LR = 0.03
N_HIDDEN = 2
ACTIVATION = F.tanh
B_INIT = -0.2   # use a bad bias constant initializer

# training data
x = np.linspace(-7, 10, N_SAMPLES)[:, np.newaxis]
noise = np.random.normal(0, 2, x.shape)
y = np.square(x) - 5 + noise

# test data
test_x = np.linspace(-7, 10, 200)[:, np.newaxis]
noise = np.random.normal(0, 2, test_x.shape)
test_y = np.square(test_x) - 5 + noise

train_x, train_y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
test_x = Variable(torch.from_numpy(test_x).float(), volatile=True)  # not for computing gradients
test_y = Variable(torch.from_numpy(test_y).float(), volatile=True)

train_dataset = Data.TensorDataset(data_tensor=train_x, target_tensor=train_y)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)
class Netme(nn.Module):
    def __init__(self):
            super(Netme, self).__init__()
              #对初始数据归一化
            self.dnn = nn.Sequential(
            nn.BatchNorm1d(1, momentum=0.5),   # input shape (1, 28, 28)
            nn.Linear(, 10),                 # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                                         # output shape (16, 28, 28)
            
            nn.BatchNorm1d(10, momentum=0.5),# activation
            nn.ReLU(),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
    def forward(self, x):
            return self.dnn(x)

netme1=Netme()
print (netme1)
exit()

class Net(nn.Module):
    def __init__(self, batch_normalization=False):
        super(Net, self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1, momentum=0.5)   # for input data

        for i in range(N_HIDDEN):               # build hidden layers and BN layers
            input_size = 1 if i == 0 else 10
            fc = nn.Linear(input_size, 10)
            setattr(self, 'fc%i' % i, fc)       # IMPORTANT set layer to the Module
            self._set_init(fc)                  # parameters initialization
            self.fcs.append(fc)
            if self.do_bn:
                bn = nn.BatchNorm1d(10, momentum=0.5)
                setattr(self, 'bn%i' % i, bn)   # IMPORTANT set layer to the Module
                self.bns.append(bn)

        self.predict = nn.Linear(10, 1)         # output layer
        self._set_init(self.predict)            # parameters initialization

    def _set_init(self, layer):
        init.normal(layer.weight, mean=0., std=.1)
        init.constant(layer.bias, B_INIT)

    def forward(self, x):
        pre_activation = [x]
        if self.do_bn: x = self.bn_input(x)     # input batch normalization
        layer_input = [x]
        for i in range(N_HIDDEN):
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn: x = self.bns[i](x)   # batch normalization
            x = ACTIVATION(x)
            layer_input.append(x)
        out = self.predict(x)
        return out, layer_input, pre_activation

nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
print(*nets)
    # print net architecture

opts = [torch.optim.Adam(net.parameters(), lr=LR) for net in nets]

loss_func = torch.nn.MSELoss()



# training
losses = [[], []]  # recode loss for two networks
for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    
    layer_inputs, pre_acts = [], []
    
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = Variable(b_x), Variable(b_y)
        for net, opt in zip(nets, opts):     # train for each network
            pred, _, _ = net(b_x)
            loss = loss_func(pred, b_y)
            opt.zero_grad()
            loss.backward()
            opt.step()    # it will also learns the parameters in Batch Normalization

            print(loss_func)

