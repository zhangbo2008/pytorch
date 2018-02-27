import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
#http://download.tensorflow.org/example_images/flower_photos.tgz
#下载完之后放到目录里面的flower文件夹里面,这种方法,子文件夹名字就是label
#很方便.
#主要就是看ImageFolder的参数
help(torchvision.transforms)


img_data = torchvision.datasets.ImageFolder('./flower_photos',
                                            transform=transforms.Compose([
                                                transforms.Scale(256),    #scale图片放缩到多少像素
                                                transforms.CenterCrop(224), #中心裁剪224像素
                                                transforms.ToTensor()])
                                            )
#从这里往下都学过了
print(len(img_data))
data_loader = torch.utils.data.DataLoader(img_data, batch_size=20,shuffle=True)
print(len(data_loader))


def show_batch(imgs):
    grid = utils.make_grid(imgs,nrow=5)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


for i, (batch_x, batch_y) in enumerate(data_loader):
    if(i<4):
        print(i, batch_x.size(), batch_y.size())
        print(batch_x)      #是20个图片
        print (batch_y)    #是20个标签
        show_batch(batch_x)
        plt.axis('off')
        plt.show()
