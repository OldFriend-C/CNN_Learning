import torchvision
from d2l import torch as d2l

def apply(img,aug,num_rows=2,num_cols=4,scale=1.5):
    Y=[aug(img) for _ in range(num_rows*num_cols)]
    d2l.show_images(Y,num_rows,num_cols,scale=scale)

d2l.set_figsize()
img=d2l.Image.open('./img/cat1.jpg')

#水平随机反转:
horizon_aug=torchvision.transforms.RandomHorizontalFlip()
#垂直随机反转:
#随机剪裁:
shape_aug=torchvision.transforms.RandomResizedCrop(
    (200,200),scale=(0.1,1),ratio=(0.5,2))
#随机改变图片亮度:
brightness_aug=torchvision.transforms.ColorJitter(
        brightness=0.5,contrast=0,saturation=0,hue=0)
#随机改变图片色调:
colorJitter_aug=torchvision.transforms.ColorJitter(
    brightness=0,contrast=0,saturation=0,hue=0.5)

#随机改变图片色调(hue)，对比度(contrast)，亮度(brightness),饱和度(saturation):
color_aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)
#多种方式组合使用
augs=torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    color_aug,shape_aug
])

