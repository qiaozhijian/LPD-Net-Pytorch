import torch
import torch.nn as nn
import config as cfg
import pynvml
import os
from torch.autograd import Variable

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# 这里的0是GPU id
ratio = 1024**2

def print_gpu():
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used / ratio
    print("used: ", used)


def funcfg(cfg):
    print(cfg.MODEL_FILENAME)

def fun():
    print_gpu()
    inner = torch.randn((44, 4096, 4096)).cuda()
    print_gpu()
    xx = torch.randn((44, 1, 4096)).cuda()
    print_gpu()
    pairwise_distance = -xx - inner
    print_gpu()
    pairwise_distance = pairwise_distance - xx.transpose(2, 1)  # [b,num,num]
    print_gpu()
    # pairwise_distance2 =  -xx - inner - xx.transpose(2, 1)  # [b,num,num]
    # print_gpu()
    # del inner
    # print_gpu()
    # print(torch.sum(pairwise_distance2-pairwise_distance))
    # return pairwise_distance

if __name__ == "__main__":
    x = Variable(torch.ones(2, 2), requires_grad=True)
    a=torch.tensor([4])
    b = torch.ones(2, 2)*8
    y = torch.cat((b, x), dim=0)
    y = y*2
    loss = torch.sum(y * a)
    # del a,b,y
    loss.backward()
    print(x.grad.data)
    # pairwise_distance = fun()
    # print_gpu()
    # pairwise_distance2 = pairwise_distance
    # print_gpu()
    # del pairwise_distance
    # print_gpu()

    
