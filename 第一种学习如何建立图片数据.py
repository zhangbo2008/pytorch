import torch
import torchvision
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
import os
mnist_test= torchvision.datasets.MNIST(
    './mnist', train=False, download=True
)
print('test set:', len(mnist_test))

f=open('mnist_test.txt','w')
for i,(img,label) in enumerate(mnist_test):
    
    #下面的路径必须提前建立好不然会报错
    if not os.path.exists('./mnist_test'):
        os.makedirs('./mnist_test')
    img_path="./mnist_test/"+str(i)+".jpg"
    
    io.imsave(img_path,img)
    
    f.write(img_path+' '+str(label)+'\n')
f.close()








from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image



def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt='mnist_test.txt', transform=transforms.ToTensor())
data_loader = DataLoader(train_data, batch_size=2,shuffle=True)

print(len(data_loader))
exit()

def show_batch(imgs):
    grid = utils.make_grid(imgs)
    print (grid) #[torch.FloatTensor of size 3x392x242],这个输出格式不重要,他只是一个图像的输出
                 #具体来扣这个grid怎么生成,是3是channel,392*242是输出的图片大小,他是imags这写个图片的组成的一个大图片
                 #使用的padding是2即,任意2个图片之见的横向和纵向距离都是2.具体可以自己算.总之不用太抠这个输出tensor
    
    
    help(plt.imshow)
    exit()
    plt.imshow(grid.numpy().transpose((1, 2, 0)))#这也是一个套路,处理rgb图像,因为上面第一个维度是3表示channel,
                                                 #这种数据转换成图片时候需要交换1,3axis,把channel放后面才能imshow
    plt.title('Batch from dataloader')


for i, (batch_x, batch_y) in enumerate(data_loader):
    if(i<4):
        print(i, batch_x.size(),batch_y.size())
        print (batch_x) #[torch.FloatTensor of size 100x3x28x28]  100:batch_size 3:channel 28*28:pic_size
        show_batch(batch_x)
        plt.axis('off')
        plt.show()
