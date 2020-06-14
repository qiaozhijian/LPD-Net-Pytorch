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

model=2
if __name__ == "__main__":
    print(1400//6*6)
    print(6*233*2%(1400//6*6)==0)
    
