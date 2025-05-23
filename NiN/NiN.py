import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
        nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.ReLU(), nn.Conv2d(out_channel, out_channel, kernel_size=1),
        nn.ReLU())


net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(3, stride=2), nn.Dropout(0.5),
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

if __name__ == '__main__':
    x = torch.rand(1, 1, 224, 224)
    for layer in net:
        x = layer(x)
        print(f'{layer.__class__.__name__},output shape: \t {x.shape}')
    lr,num_epoches,batch_size=0.1,10,128
    train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)
    d2l.train_ch6(net,train_iter,test_iter,num_epoches,lr,d2l.try_gpu())
    d2l.plt.show()