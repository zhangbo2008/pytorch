#先写pca的变换和逆变换的到的pca白化.
#coding=utf-8
from numpy import *

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''
def eigValPct(eigVals,percentage):
    sortArray=sort(eigVals) #使用numpy中的sort()对特征值按照从小到大排序
    
    
    sortArray=sortArray[::-1] #特征值从大到小排序
    
    arraySum=sum(sortArray) #数据全部的方差arraySum
    #percentage 表示的是降为到所有方差和的多少.也就是保留数据的多少波动性.越大保留越多
    #比如percentage写0.99那么就是num返回2 ,如果写0.7就返回1
    tempSum=0
    num=0
    for i in sortArray:
        tempSum+=i
        num+=1
        if tempSum>=arraySum*percentage:
            return num
import numpy as np
'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''
def pca(dataMat,percentage=0.9):
    meanVals=mean(dataMat,axis=0)  #对每一列求平均值，因为协方差的计算中需要减去均值
    meanRemoved=dataMat-meanVals
    
    covMat=cov(meanRemoved,rowvar=0)  #cov()计算方差
      #是一个4*4的矩阵  即feature_size*feature_size
    eigVals,eigVects=linalg.eig(mat(covMat))  #利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    
    U,S,V = np.linalg.svd(mat(covMat)) #奇异分解
    #这个U就是上面的eigVals斯密特正交化之后的矩阵.
    
    
    k=eigValPct(eigVals,percentage) #要达到方差的百分比percentage，需要前k个向量
    
    
    
    eigValInd=argsort(eigVals)  #对特征值eigVals从小到大排序
    
    eigValInd=eigValInd[:-(k+1):-1] #从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    
    global vect
    vect=eigVals[eigValInd]
    #vect就是我们要变换的特征值.
    
    
    
    redEigVects=eigVects[:,eigValInd]   #返回排序后特征值对应的特征向量redEigVects（主成分）
    #投影就是直接矩阵乘法,两个向量做内机就是做投影,归结到矩阵就是矩阵乘法.
    #其实不是投影,而是一个忽略了特征向量摸长的投影,所以这个做完了得到的是一个投影之后在主方向上再伸缩一下.
    #这里周志华树里面给的特征向量组成的这个矩阵是u矩阵的.


    #重要性质:一个n行n列的实对称矩阵一定可以找到n个单位正交特征向量,证明显然.
    #但是问题是直接通过numpy里面的eig得到的是正交的向量吗?只需要看下面的这个是不是除了对角线上全都是0
    #经过实验还真不是.numpy给的是随便给的一个.
##    print (redEigVects*redEigVects.T)
    
    
    lowDDataMat=meanRemoved*redEigVects #将原始数据投影到主成分上得到新的低维数据lowDDataMat
##    print (redEigVects*redEigVects.T)  #特征向量矩阵不是u矩阵
    reconMat=(lowDDataMat*redEigVects.T)+meanVals   #得到重构数据reconMat?没懂怎么重构的
    #据我分析这里面乘以专职也就是一个近似的逆,然后得到后也是忽略方向的伸缩
    #貌似这里面的专职就能达到乘以逆的效果,需要证明一下.这个证明也是显然的.

    #总结一下就是直接做*redEigVects*redEigVects.T就达到了降为然后再你回去的效果.确实很方便.

    #下面3行写zca白化
    #其实只需要在pca里面加上除以标准差str这里面用xishu来表示即可.上面封装太多.还是用自己写的第一种方法才行.
    epsilon = 0.1 

    xishu=(1.0/np.sqrt(np.diag(vect) + epsilon))#这就是白化需要的系数

    
    baihua=lowDDataMat*xishu*redEigVects.T

    
    return lowDDataMat,reconMat,baihua
a=pca(array([[9,43,4343,-88],[9,4324,43,-8],[4,5,6,70],[1,2,300,4000],[1,2,300,46],[4,5,60,70]]))
print (a[1])#这个输出的就是pca之后再变回去的数据
print (a[2])#这个是白化的,注意白化处理的数据必须是归一化之后的.mean为0的,否则没啥意义.
a=array([[9,43,4343,-88],[9,4324,43,-8],[4,5,6,70],[1,2,300,4000],[1,2,300,46],[4,5,60,70]])



#下面是用库包来实现的,更少代码~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from sklearn import decomposition
from sklearn import datasets
meanVals=mean(a,axis=0)
pca = decomposition.PCA(n_components=3)
pca.fit(a)
a = pca.transform(a)


a = np.matrix(a)
b = np.matrix(pca.components_)
c = a * b

c+=meanVals
print (c) #从结果看到完全一样.































