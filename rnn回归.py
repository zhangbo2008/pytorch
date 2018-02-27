"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
numpy
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 10      # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate

### show data
##steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
##print (steps),'其实直接逗号然后两个引号也可以达到注释的效果'  
##x_np =1/( (steps)*(steps) )   # float32 for converting torch FloatTensor
##y_np = np.cos(steps)+np.sin(steps)*np.sin(steps)
##plt.plot(steps, y_np, 'r-', label='target (cos)')
##plt.plot(steps, x_np, 'b-', label='input (sin)')
##plt.legend(loc='best')

'''用2层神经网络来逼近,
第一层是rnn带记忆的来跑,第二层是放射,来做放缩和平移图像,从第二层来看显然会学的很完美,因为cos和sin就
差一个平移,第一层带记忆所以能学习更复杂的曲线,把sin改成其他函数来跑,比如我直接用y=x来逼近,
直接x_np=steps,跑出来效果一样好,原因就是带记忆,学习力很强,这里面后来我随便改x_np和y_np
都能跑'''



'''话说这个.py里面的batch好像没用到'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,     # rnn hidden unit
            num_layers=1,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        #x 1*10*1
        r_out, h_state = self.rnn(x, h_state) #r_out  1*10*32    h_state 1*1*32
        
        
        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :])) 
        #stack把一个列表展平
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # return outs, h_state

rnn = RNN()


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()
#给初始状态一个初值,给None就行
h_state = None      # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(60):#每一个step做一个预测,画一个点
    start, end = step * np.pi, (step+1)*np.pi   # time range每次用一个time range来做cell预测
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    #下面的rnn里面的参数不止有x还有上个状态h_state所以他有记忆功能,跟上一个状态也有关.
    prediction, h_state = rnn(x, h_state)   # rnn output
    
    # !! next step is important !!
    h_state = Variable(h_state.data)        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw(),'不太懂这个动态模式画图的draw有毛用'
    plt.pause(0.05)

plt.ioff()
plt.show()
