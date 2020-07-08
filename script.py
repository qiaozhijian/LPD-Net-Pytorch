import torch
import torch.nn as nn
import torch.nn.functional as F
# import config as cfg
import pynvml
import os
# from torch.autograd import Variable
# import numpy as np
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# 这里的0是GPU id
ratio = 1024**2

def print_gpu():
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used / ratio
    print("used: ", used)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, (3,4), padding=False)  # 输入通道数为1，输出通道数为6
        self.conv2 = nn.Conv2d(6, 16, 5, padding=True)  # 输入通道数为6，输出通道数为16
        self.fc1 = nn.Linear(16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入x -> conv1 -> relu -> 2x2窗口的最大池化
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        # 输入x -> conv2 -> relu -> 2x2窗口的最大池化
        x = self.conv2(x)
        print(x.shape)
        x = F.relu(x)
        print(x.shape)
        x = F.max_pool2d(x, 2)
        print(x.shape)
        x = x.permute(0, 2, 3, 1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = self.fc3(x)
        print(x.shape)
        return x

def get_learning_rate(epoch):
    learning_rate = 0.001*(0.922680834591**epoch)
    learning_rate = max(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate
if __name__ == '__main__':

    a=get_learning_rate(10)
    print(a)