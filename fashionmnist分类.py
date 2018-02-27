import os
from skimage import io
#skimage是一个sklearn里面读取图像的
import torchvision.datasets.mnist as mnist
#这里面.表示这个py文件所在的目录
root="./fashion/"
#mnist.read_image_file是里面的一个api,读取这种图片文件
train_set = (
    mnist.read_image_file(os.path.join(root, 'train-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 'train-labels-idx1-ubyte'))
        )
test_set = (
    mnist.read_image_file(os.path.join(root, 't10k-images-idx3-ubyte')),
    mnist.read_label_file(os.path.join(root, 't10k-labels-idx1-ubyte'))
        )
print("training set :",train_set[0].size())
print("test set :",test_set[0].size())
#下面这行把test和train都转换成图片然后存到fashion这里面的2个文件夹里面
#不知道这个操作有毛卵用
#然后label都存到train.txt和test.txt这2个文件里面
def convert_to_img(train=True):
    if(train):
        f=open(root+'train.txt','w')
        data_path=root+'/train/'
        if(not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(train_set[0],train_set[1])):
            img_path=data_path+str(i)+'.jpg'
            #先转换成numpy然后保存到img_path这个文件里面
            io.imsave(img_path,img.numpy())
            f.write(img_path+' '+str(label)+'\n')
        f.close()
    else:
        f = open(root + 'test.txt', 'w')
        data_path = root + '/test/'
        if (not os.path.exists(data_path)):
            os.makedirs(data_path)
        for i, (img,label) in enumerate(zip(test_set[0],test_set[1])):
            img_path = data_path+ str(i) + '.jpg'
            io.imsave(img_path, img.numpy())
            f.write(img_path + ' ' + str(label) + '\n')
        f.close()
if not os.path.exists(root+'train/'):
    #如果没有这个文件夹再跑train,跑过了就别跑了
    
    convert_to_img(True)
    convert_to_img(False)
else:
    print ('没有跑converttoimage')

    
#下面开始正式的算法,上面算作清晰数据吧.
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
root="./fashion"

# -----------------ready the dataset--------------------------

'''模式“RGB”为24位彩色图像，它的每个像素用24个bit表示，分别表示红色、绿色和蓝色三个通道。

在PIL中，对于彩色图像，open后都会转换为“RGB”模式，然后该模式可以转换
为其他模式，比如“1”、“L”、“P”和“RGBA”，这几种模式也可以转换为“RGB”模式。

1、 模式“1”转换为模式“RGB”

模式“RGB”转换为模式“1”以后，像素点变成黑白两种点，要么是0，要么是255。而从模式“1”转换
成“RGB”时，“RGB”的三个通道都是模式“1”的像素值的拷贝。'''
#这里面强制把图片从L变成rgb 的3个
#屌丝用print() 来调试程序,后面接一个exit()就能强制退出,然后看print结果了.
#可能是下面的网络写的是处理rgb的所以他这里转换成rgb了.
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
            #imgs来保存图片的路径和图片的正确标签
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        
        #上面写好了load函数,所以这里面读取之后的图片是rgb的
        img = self.loader(fn)
        if self.transform is not None:
            #就是totensor
            img = self.transform(img)
        return img,label
        #这里get得到的是图片不是矩阵.因为上面已经io.imsave了.

    def __len__(self):
        return len(self.imgs)

train_data=MyDataset(txt=root+'train.txt', transform=transforms.ToTensor())


test_data=MyDataset(txt=root+'test.txt', transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)


#-----------------create the Net and training------------------------

class Net(torch.nn.Module):
    def __init__(self):
        #super这个函数表示括号里面第一个参数的父类.并且第一个参数随便可以修改,可以与他所在的类不同.
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),   # input, output ,kernal ,stride ,padding
                                               #因为图像是rgb所以这里面我们input=3,output 随意
                                               #kenal 也随意,就是滤波器是3*3的,这个东西越小越精确
                                               #stride 基本取1,也是越小越好
                                               #padding 填充 因为我们kenal是3所以这个数要写1才能保证数据,
                                               #这个具体问题算一下就行,简单的padding计算,还有padding填充的都是数字0
                                               #好像没给其他的填充方式...还有一种常用的填充方式是就近填充,不知道
                                               #pytorch如何写.懂了,其实不用pytorch写,因为填充这东西可以用numpy来写
                                            #比如你要临近填充,你可以手动填充后,然后进入Conv2d,然后设置padding=0就完了.
                                             #Conv1d,Conv3d不懂,以后遇到再学吧.
                                        #注意cnn处理channel的方式就是把他们加起来再加bias.所以rgb处理完肯定是黑白了.
#ps:
#            CNN中的卷积和池化的边界问题一般怎么处理
#有两种方式
##
##1.不处理边界。5*5的src通过3*3的卷积核，步长1时，就变成了3*3的output
##2.在边界补0。5*5的src，补1圈0变成7*7，通过3*3的卷积核，步长1时，就变成了5*5的output，即原尺寸
##通常卷积可采用1,2.但是池化一般只采用1 !!!!!!!!!!!!!!!!!!!!所以下面最后一次的图形变成了3*3而不是4*4
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))          #这一行跑完图像变成14*14
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)           #这行跑完图像变成 7*7
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)           #这行跑完图像变成 3*3!
                                            #为什么不是4*4?原因就是pooling的边界处理.
        )
        self.dense = torch.nn.Sequential(     #这层是控制输出的,把输入的图片都占成1维的进行普通dnn训练来输出最后10神经.
            torch.nn.Linear(64 * 3 * 3, 128), #听说还能加一种bn层,batch normalization 可以更快的学习.
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
##        print (conv1_out)           结果是 batch*32*14*14   第一个位置batch,第二个位置是cout_size
                                    #                         第三个和第四个位置是图片大小14*14
     
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1) #conv3_out.size(0)就是batch,而后面的-1也就是 64*3*3
        out = self.dense(res)
        return out


model = Net()
print(model)
#下面2行是套路,一般不用改,从整体上看，Adam目前是最好的选择
optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()
#开始for 训练和输出预测accuracy
for epoch in range(10):  #epoch代表第几轮的全数据.
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        #下面这一行需要写?必须是Variable才行?,试过了,必须这么写
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data[0]
#这里面用max然后取[1],直接找到每一行最大的概率是第几个index :torch.max(test_output, 1)[1]
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
##      print(train_correct) 得到的居然是一个列表,(pred == batch_y)表示获得一个1,0组成的列表再取sum得到一个1维tensor
#所以用train_correct.data[0]来提取这个数
        train_acc += train_correct.data[0]
        #上面就是前向传播
        #下面3行就是反向传播的套路3句话,一般不用改.从改没见到改过这3句话....
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #每一个batch打印一次,不然他写的每一次epoch打印一次太慢了,基本跑不动,这里面train_loss没卵用,他统计的是一个epoch的
        #不是每一个batch的.
        #然后打印这个batch的成绩,这个print写的太风骚自己不太会写
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

    # evaluation--------------------------------
    #上面一个poch跑完,也就训练好了,下面开始test
    #.eval()表示把网络切换成测试模式..不是太懂
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data[0]
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data[0]
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))
