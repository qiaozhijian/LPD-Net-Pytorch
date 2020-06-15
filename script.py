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

def change(a):
    a = "dae"


if __name__ == "__main__":
    a= "{0:145,1:236}"
    change(a)
    print(a)