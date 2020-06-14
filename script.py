import torch
import torch.nn as nn
import config as cfg
import pynvml
import os
from torch.autograd import Variable
import numpy as np

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# 这里的0是GPU id
ratio = 1024**2

def print_gpu():
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used / ratio
    print("used: ", used)

model=2
if __name__ == "__main__":
    base=[]
    for i in range(20):
        a=np.random.randn(4096,3)
        base.append(a)
    base=np.asarray(base).reshape(-1,4096,3)
    b=base[[0,2,15]]
    print(b.shape)
    
