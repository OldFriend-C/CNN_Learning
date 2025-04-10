import torch
import math
import random
import numpy as np
from d2l import torch as d2l
def synthetic_data(w,b,num_examples):
    """生成y=Xw+b+噪音"""
    X=torch.normal(0,1,(num_examples,len(w)))
    y=torch.matmul(X,w)+b
    y+=torch.normal(0, 0.01, y.shape)
    return  X,y.reshape(-1,1)

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indexs=list(range(num_examples))
    random.shuffle(indexs)
    for i in range(0,num_examples,batch_size):
        batch_index=torch.tensor(indexs[i:min(i+batch_size,num_examples)])
        yield features[batch_index],labels[batch_index]

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#真实的权重和偏差
true_w=torch.tensor([2,-3.4])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)

#设置批量大小为10
batch_size=10

#1.初始化参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#这里设置了学习率(lr)和训练次数(num_epochs)
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        #2.每个迭代周期梯度优化
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        #3.每个迭代周期更新参数
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    #每个周期输出结果
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)  #整个数据集的损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差: {true_b - b}')



