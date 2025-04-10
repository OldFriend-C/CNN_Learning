import torch
from torch import nn
from d2l import torch as d2l


"""调用nn模组里面的BatchNorm可以进行批量归一化"""
leNet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6),
    nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16),
    nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),nn.Linear(256, 120),nn.BatchNorm1d(120),
    nn.Sigmoid(),nn.Linear(120, 84), nn.BatchNorm1d(84),
    nn.Sigmoid(),nn.Linear(84, 10)
)