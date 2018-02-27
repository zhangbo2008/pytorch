import torch
import torchvision
import numpy as np
x=np.arange(18).reshape(2,3,3)
x=torch.from_numpy(x)   #numpy to torch tensor
x=x.view(x.size(0), -1)
print (x)
x = x.numpy() # torch tensor to numpy
print (x)
